"""Local adapter for the CogWorks platform-test submission.

This module mirrors, locally and offline, the contract path the CogWorks
portal exercises on a submission repository: load, instantiate, embed,
prepare, search, and score -- and refuse a submission whose embeddings
violate the declared shapes, before any scoring happens.

It is a test-support component of THIS repository. It is not the hosted
platform: behaviour verified through this adapter is evidence about the
pinned submission code, never about hosted platform behaviour.

Pinned local contract (an assumption recorded in README "Limits"):
- ``embed_text(captions)`` returns a ``numpy.ndarray`` of shape
  ``(len(captions), 64)`` with a floating dtype. Array-likes are refused:
  the pinned submission returns ndarrays.
- ``embed_images(descriptors)`` returns an ndarray of shape
  ``(len(descriptors), 64)``.
- ``prepare_database(image_ids, descriptors)`` stores ``int(image_id)`` per
  id, in order, and requires ``(n, 512)`` descriptors (the projection layer
  is fixed at 512 inputs).
- ``search(query, k)`` returns a ``list[int]`` of length ``min(k, n_ids)``
  for integer ``k >= 0``, ids drawn from the prepared ids, in stable
  descending score order.

Every refusal raises :class:`SubmissionContractError` carrying a single-line
``note`` suitable for rendering on a run report. Notes are deterministic:
same violation, same text, no exception-class noise, no tracebacks.
"""

from __future__ import annotations

import importlib.util
import numbers
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Fixed submission-side dimensions (see submission.py on main).
DIMENSIONS = 64
DESCRIPTOR_DIM = 512


class SubmissionContractError(Exception):
    """A submission violated the local contract before scoring could run.

    ``note`` is the deterministic, single-line diagnostic intended for a run
    report; ``str(exc)`` equals ``note``.
    """

    def __init__(self, note: str):
        self.note = note
        super().__init__(note)


# ---------------------------------------------------------------------------
# Loading (mirrors the portal "fetch" stage: load by explicit path)
# ---------------------------------------------------------------------------

def load_submission(path: str | Path, module_name: str) -> Any:
    """Import a submission module from an explicit file path.

    The module is loaded under ``module_name`` without touching
    ``sys.path`` and without polluting the ``sys.modules`` entry
    ``"submission"``; callers that want isolation pass unique module names.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise SubmissionContractError(
            f"submission file not found at {path}; run refused before scoring"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SubmissionContractError(
            f"could not build an import spec for {path}; run refused before scoring"
        )
    module = importlib.util.module_from_spec(spec)
    # Deliberately NOT inserted into sys.modules: import isolation.
    spec.loader.exec_module(module)
    return module


def instantiate(module: Any, resources: dict | None = None) -> Any:
    """Build the submission object via the pinned factory entry point."""
    factory = getattr(module, "create_submission", None)
    if not callable(factory):
        raise SubmissionContractError(
            "submission module exposes no callable create_submission(resources); "
            "run refused before scoring"
        )
    return factory(dict(resources or {}))


# ---------------------------------------------------------------------------
# embed_text contract
# ---------------------------------------------------------------------------

def embed_text_checked(submission: Any, captions: Sequence[str]) -> np.ndarray:
    """Call embed_text and enforce the (rows, DIMENSIONS) matrix contract."""
    if not isinstance(captions, (list, tuple)):
        captions = list(captions)
    out = submission.embed_text(list(captions))
    method = "embed_text"
    n = len(captions)
    if not isinstance(out, np.ndarray):
        raise SubmissionContractError(
            f"{method} returned {type(out).__name__}, expected a numpy ndarray of "
            f"shape ({n}, {DIMENSIONS}); run refused before scoring"
        )
    if out.ndim == 1:
        width = int(out.shape[0])
        raise SubmissionContractError(
            f"{method} returned shape {tuple(out.shape)}, expected ({n}, {DIMENSIONS}) "
            f"for {n} captions; ndim=1 dtype={out.dtype} width={width}; run refused "
            f"before scoring: a per-caption retry cannot rescue a 1-D return, and "
            f"width {width} is below the fixed embedding width {DIMENSIONS}"
        )
    if out.ndim != 2:
        raise SubmissionContractError(
            f"{method} returned shape {tuple(out.shape)}, expected ({n}, {DIMENSIONS}) "
            f"for {n} captions; run refused before scoring: expected exactly 2 "
            f"dimensions, got {out.ndim}"
        )
    rows, width = out.shape
    if width != DIMENSIONS:
        raise SubmissionContractError(
            f"{method} returned shape {tuple(out.shape)}, expected ({n}, {DIMENSIONS}) "
            f"for {n} captions; width {width} does not match the fixed embedding "
            f"width {DIMENSIONS}; run refused before scoring"
        )
    if rows != n:
        raise SubmissionContractError(
            f"{method} returned shape {tuple(out.shape)}, expected ({n}, {DIMENSIONS}); "
            f"row count {rows} does not match {n} captions; run refused before scoring"
        )
    if not np.issubdtype(out.dtype, np.floating):
        raise SubmissionContractError(
            f"{method} returned dtype {out.dtype}, expected a floating dtype; "
            f"run refused before scoring"
        )
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# embed_images contract
# ---------------------------------------------------------------------------

def embed_images_checked(submission: Any, descriptors: np.ndarray) -> np.ndarray:
    """Call embed_images and enforce the (rows, DIMENSIONS) matrix contract."""
    out = submission.embed_images(descriptors)
    method = "embed_images"
    n = int(descriptors.shape[0])
    if not isinstance(out, np.ndarray):
        raise SubmissionContractError(
            f"{method} returned {type(out).__name__}, expected a numpy ndarray of "
            f"shape ({n}, {DIMENSIONS}); run refused before scoring"
        )
    if out.ndim != 2 or out.shape[1] != DIMENSIONS or out.shape[0] != n:
        raise SubmissionContractError(
            f"{method} returned shape {tuple(out.shape)}, expected ({n}, {DIMENSIONS}) "
            f"for {n} descriptors; run refused before scoring"
        )
    if not np.issubdtype(out.dtype, np.floating):
        raise SubmissionContractError(
            f"{method} returned dtype {out.dtype}, expected a floating dtype; "
            f"run refused before scoring"
        )
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# prepare stage (mirrors the portal "prepare" step)
# ---------------------------------------------------------------------------

def prepare_checked(submission: Any, image_ids: Iterable[int],
                    descriptors: np.ndarray) -> list[int]:
    """Validate prepare inputs, then call prepare_database.

    Input validation happens here, in the adapter, so a bad descriptor
    matrix produces a one-line note instead of a raw numpy traceback.
    """
    ids = [int(i) for i in image_ids]
    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2:
        raise SubmissionContractError(
            f"prepare_database descriptors have shape {tuple(descriptors.shape)}, "
            f"expected (n_images, {DESCRIPTOR_DIM}); run refused before scoring"
        )
    if descriptors.shape[1] != DESCRIPTOR_DIM:
        raise SubmissionContractError(
            f"prepare_database descriptors have shape {tuple(descriptors.shape)}, "
            f"expected (n_images, {DESCRIPTOR_DIM}); descriptor width "
            f"{descriptors.shape[1]} does not match the fixed descriptor width "
            f"{DESCRIPTOR_DIM}; run refused before scoring"
        )
    if descriptors.shape[0] != len(ids):
        raise SubmissionContractError(
            f"prepare_database got {len(ids)} image ids but "
            f"{descriptors.shape[0]} descriptor rows; run refused before scoring"
        )
    submission.prepare_database(ids, descriptors.astype(np.float32, copy=False))
    return ids


# ---------------------------------------------------------------------------
# search contract
# ---------------------------------------------------------------------------

def search_checked(submission: Any, query: str, k: int,
                   n_ids: int) -> list[int]:
    """Call search and enforce the id-list contract."""
    if not isinstance(k, numbers.Integral) or isinstance(k, bool) or k < 0:
        raise SubmissionContractError(
            f"search called with k={k!r}, expected a non-negative integer; "
            f"run refused before scoring"
        )
    out = submission.search(query, int(k))
    expected = min(int(k), n_ids)
    if not isinstance(out, list):
        raise SubmissionContractError(
            f"search returned {type(out).__name__}, expected a list of image ids; "
            f"run refused before scoring"
        )
    if len(out) != expected:
        raise SubmissionContractError(
            f"search returned {len(out)} ids, expected {expected} for k={int(k)} "
            f"over {n_ids} prepared images; run refused before scoring"
        )
    for el in out:
        if isinstance(el, bool) or not isinstance(el, numbers.Integral):
            raise SubmissionContractError(
                f"search returned a non-integer id {el!r} of type "
                f"{type(el).__name__}; run refused before scoring"
            )
    return [int(el) for el in out]


# ---------------------------------------------------------------------------
# Local benchmark pipeline (fetch -> prepare -> evaluate -> score -> report)
# ---------------------------------------------------------------------------

def run_local_benchmark(submission: Any, image_ids: Sequence[int],
                        descriptors: np.ndarray, queries: Sequence[str],
                        k: int) -> dict:
    """Run the offline analog of the platform pipeline and return results.

    Contract refusals propagate as :class:`SubmissionContractError`; a clean
    run returns embeddings, top-k id lists, and the score-order matrix.
    """
    ids = prepare_checked(submission, image_ids, descriptors)
    text_matrix = embed_text_checked(submission, list(queries))
    top_k = [search_checked(submission, q, k, len(ids)) for q in queries]
    return {
        "image_ids": ids,
        "text_matrix": text_matrix,
        "top_k": top_k,
        "k": int(k),
    }


def topk_hits(top_k: Sequence[Sequence[int]],
              correct: Sequence[int]) -> int:
    """Number of queries whose correct image appears in their top-k list."""
    return sum(1 for row, c in zip(top_k, correct) if c in row)


def topk_family_overlap(top_k: Sequence[Sequence[int]],
                        correct: Sequence[int],
                        family_lookup) -> int:
    """Total overlap between returned top-k ids and each query's family.

    ``family_lookup`` maps a correct image ID (the same domain as the ids
    returned by search) to an iterable of ids in its family.
    """
    total = 0
    for row, c in zip(top_k, correct):
        family = set(family_lookup(c))
        total += len({i for i in row if i in family})
    return total
