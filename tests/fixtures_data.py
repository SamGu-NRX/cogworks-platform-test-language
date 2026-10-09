"""Declared fixture distribution and pinned oracle constants.

This file is the SINGLE SOURCE OF TRUTH for every expected value and score
band used by the offline tests. Test files import from here; they must not
restate band numbers, so a band can only change by editing this file, where
the derivation sits next to the constant.

Everything here is SYNTHETIC. Nothing in this file encodes real course data,
real student records, or any hosted-platform measurement.

Score-band policy: bands are derived analytically from the declared
distribution below and pinned at full precision. A band is never widened
because an observed run landed outside it -- an out-of-band run is a
finding to investigate (fixture bug, contract break, or a derivation
error), not an inconvenience to absorb.

Score-value policy: tiny-fixture score values are pinned as exact float64
reprs of the float32 pipeline output and compared with rtol=1e-6, because
BLAS reduction order may vary across numpy builds. Top-k ORDERINGS are
pinned exactly and must match bit-for-bit.
"""

import hashlib
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# Fixed submission-side dimensions (mirror of submission.py constants)
# ---------------------------------------------------------------------------

DIMENSIONS = 64
DESCRIPTOR_DIM = 512
PROJECTION_SEED = 20261002  # np.random.RandomState seed inside submission.py

# ---------------------------------------------------------------------------
# Pinned source identities
# ---------------------------------------------------------------------------
# The submission is a frozen platform fixture. Any change to either file is a
# contract change and must be made deliberately, by updating these pins with
# a note explaining the new provenance -- never silently.

MAIN_SUBMISSION_SHA256 = "f0fc788721af98176c7b3d3629b64b8adfc655825b5de1d48734ea442edaa8f2"
MALFORMED_BLOB_SHA = "1784a8b79d6cea85d836ee58a49e8000b2728621"
MALFORMED_SUBMISSION_SHA256 = "2d6c9a7610913509118bc96f57b851ef8af51fb9e30c98a1b240372a1d6a7c7c"
MALFORMED_SOURCE_COMMIT = "4c6d54d5d42ba85436e43a579acc452bca6570d9"

def sha256_file(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()

# ---------------------------------------------------------------------------
# Tokenizer / bucket pins (crc32(word) % 64, words lowercased by the regex)
# ---------------------------------------------------------------------------

BUCKET_PIN = {
    "dog": 61, "cat": 40, "red": 15, "run": 0, "fast": 4,
    "blue": 52, "sky": 47, "hat": 9, "sun": 55, "car": 29,
    "tree": 28, "house": 29, "night": 23, "day": 16, "jump": 22,
    "walk": 21, "quiet": 36, "loud": 8, "city": 52, "field": 24,
    "moon": 40, "star": 20, "rain": 45, "snow": 51, "wind": 62,
    "fire": 15, "gold": 17, "silver": 27, "king": 35, "queen": 58,
    "crown": 36, "sword": 15, "river": 43, "mountain": 60,
    "valley": 7, "forest": 52,
}

def bucket(word: str) -> int:
    return zlib.crc32(word.encode("utf-8")) % DIMENSIONS

# Deliberate CRC-32 bucket aliasing: distinct words sharing a bucket. The
# oracle must treat word counts as bucket counts, not as an injective map.
ALIAS_PAIRS = [("blue", "city"), ("car", "house"), ("fire", "sword"), ("cat", "moon")]

# Tokenizer pins: regex [a-z0-9]+ on lowercased text.
TOKENIZER_PIN = [
    ("Dog, DOG! d0g", ["dog", "dog", "d0g"]),
    ("co-oper", ["co", "oper"]),
    ("42", ["42"]),
    ("!!!", []),
    ("", []),
]

# ---------------------------------------------------------------------------
# Tiny caption-vector oracle (exact)
# ---------------------------------------------------------------------------

def tiny_vector(text: str) -> np.ndarray:
    """Reference reimplementation of the bucket-count embedder."""
    import re
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[bucket(word)] += 1.0
    return vector

TINY_CAPTION_PINS = [
    # (text, {bucket: count}, exact float32 norm as float64)
    ("dog dog cat", {40: 1.0, 61: 2.0}, 5 ** 0.5),
    ("cat", {40: 1.0}, 1.0),
    ("dog", {61: 1.0}, 1.0),
    ("blue city", {52: 2.0}, 2.0),
    ("!!!", {}, 0.0),
    ("", {}, 0.0),
]

# Exact unit-norm cosine values between tiny captions.
COS_PIN_DOGDOG_CAT = 1.0 / 5 ** 0.5   # dog dog cat vs cat  -> 1/sqrt(5)
COS_PIN_DOGDOG_DOG = 2.0 / 5 ** 0.5   # dog dog cat vs dog  -> 2/sqrt(5)

# ---------------------------------------------------------------------------
# Tiny search oracle: 4-image database
# ---------------------------------------------------------------------------
# Descriptors: rows are one-hot basis vectors e0, e1, e2 and a zero row, so
# embed_images(D) is exactly projection rows P[0], P[1], P[2] and a zero row.
# image ids chosen to be distinctive.

TINY_IMAGE_IDS = [101, 202, 303, 404]

def tiny_descriptors() -> np.ndarray:
    descriptors = np.zeros((4, DESCRIPTOR_DIM), dtype=np.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    descriptors[2, 2] = 1.0
    return descriptors

# Exact expected top-3 id lists. Note id 404 (the zero descriptor) ranks
# first for any nonzero query with only negative scores: chance behaviour,
# made exact. For the all-zero query the stable argsort returns stored order
# (101, 202, 303) and 404 -- tied at 0.0 -- sorts last.
TINY_TOP3_PIN = {
    "dog": [404, 101, 202],
    "cat": [404, 101, 303],
    "!!!": [101, 202, 303],
    "blue city": [101, 404, 202],
    "dog cat": [404, 101, 202],
}

# Exact expected full score vectors [id 101, 202, 303, 404], pinned to
# 1e-6 (see score-value policy above).
TINY_SCORES_PIN = {
    "dog": [-0.0457705557346344, -0.08582331240177155, -0.34593549370765686, 0.0],
    "cat": [-0.12416035681962967, -0.18309128284454346, -0.12999269366264343, 0.0],
    "blue city": [0.07837827503681183, -0.14607036113739014, -0.1761755794286728, 0.0],
}

# Exact projection samples (seed PROJECTION_SEED, standard_normal((512, 64)),
# .astype(float32)) -- pins the fixed projection without re-deriving it.
PROJECTION_SAMPLES_PIN = {
    (0, 0): -0.8382928967475891,
    (0, 1): 1.1019701957702637,
    (0, 2): 0.2801549434661865,
    (0, 3): 0.6757977604866028,
    (511, 0): 1.1005852222442627,
    (511, 1): 0.9043909311294556,
}

def reference_projection() -> np.ndarray:
    generator = np.random.RandomState(PROJECTION_SEED)
    return generator.standard_normal((DESCRIPTOR_DIM, DIMENSIONS)).astype(np.float32)

# ---------------------------------------------------------------------------
# Balanced fixture family (the declared distribution for the chance bands)
# ---------------------------------------------------------------------------
# Declared distribution (all randomness is fixed-seed, no host entropy):
# - n_images = 128; descriptors i.i.d. N(0,1) in R^512, float32, from
#   RandomState(DESCRIPTOR_SEED). Caption words are independent of the
#   descriptors, and the projection (fixed seed, shared with the submission)
#   is independent of caption content. Text lives in a bucket-count basis,
#   images in the range of a random projection of independent Gaussians:
#   query-to-image cosines are therefore (approximately) i.i.d. continuous
#   scores with no informative ordering, i.e. every ranking of the correct
#   image is equally likely marginally.
# - 8 word-families of 4 words; image i belongs to family i // 16, so each
#   family holds 16 images (balanced sizes).
# - caption(i) = family[i//16][i%4] + " " + family[i//16][(i+1)%4]:
#   every word in a family appears in exactly 8 of its 16 captions
#   (4 as first word, 4 as second word).
# - query(i) uses the same family: family words [(i%4) and (i+3)%4], so
#   queries are on-distribution captions, still independent of descriptors.
# - m = N_QUERIES = 128 queries, exactly one query per image; the correct
#   image for query i is image i itself.
# - k = K = 10.

N_IMAGES = 128
N_FAMILIES = 8
IMAGES_PER_FAMILY = 16
DESCRIPTOR_SEED = 20261028
K = 10
N_QUERIES = 128

FAMILIES = [
    ["dog", "cat", "red", "fast"],
    ["blue", "sky", "hat", "sun"],
    ["car", "tree", "house", "night"],
    ["day", "jump", "walk", "quiet"],
    ["moon", "star", "rain", "snow"],
    ["wind", "fire", "gold", "silver"],
    ["king", "queen", "crown", "sword"],
    ["river", "mountain", "valley", "forest"],
]

def family_of(index: int) -> int:
    return index // IMAGES_PER_FAMILY

def balanced_captions() -> list[str]:
    captions = []
    for i in range(N_IMAGES):
        words = FAMILIES[family_of(i)]
        captions.append(f"{words[i % 4]} {words[(i + 1) % 4]}")
    return captions

def balanced_queries() -> list[str]:
    queries = []
    for i in range(N_IMAGES):
        words = FAMILIES[family_of(i)]
        queries.append(f"{words[i % 4]} {words[(i + 3) % 4]}")
    return queries

def balanced_descriptors() -> np.ndarray:
    generator = np.random.RandomState(DESCRIPTOR_SEED)
    return generator.standard_normal((N_IMAGES, DESCRIPTOR_DIM)).astype(np.float32)

def balanced_image_ids() -> list[int]:
    return list(range(1000, 1000 + N_IMAGES))

def index_of_id(image_id: int) -> int:
    """Map a balanced-fixture image id (1000-based) to its 0-based index."""
    return image_id - 1000

def family_id_set(image_id: int) -> set[int]:
    """Ids of the IMAGES_PER_FAMILY images sharing image_id's word-family.

    Takes the IMAGE ID (as returned by search), not the 0-based index.
    """
    f = family_of(index_of_id(image_id))
    return {1000 + IMAGES_PER_FAMILY * f + j for j in range(IMAGES_PER_FAMILY)}

# Analytic expectation 1 -- HITS: P(correct image in a random top-k) = k/n
# under exchangeability, so over m queries the expected hit count is
# m*k/n = 10.0 with binomial sd sqrt(m*(k/n)*(1-k/n)) = 3.0362394503727796.
# Band: |observed - 10.0| <= 3 * sd. Independence across queries is an
# approximation (queries share the fixed image pool and projection);
# justified because caption content is independent of descriptors. The
# band's role is to catch gross contract breaks (e.g. leakage pushing hits
# far above chance), not to bound tail risk.
HITS_EXPECTED = 10.0
HITS_SD = 3.0362394503727796
HITS_SIGMA = 3.0

# Analytic expectation 2 -- FAMILY OVERLAP: overlap of a returned top-k list
# with the query's 16-image family, under the null, is
# Hypergeometric(N=n_images, K=16, draws=k): mean per query k*16/n = 1.25,
# var per query k*(16/n)*(1-16/n)*(n-k)/(n-1), so over m queries:
# expected total = m*k*16/n = 160.0, sd = 11.40520671261509.
OVERLAP_EXPECTED = 160.0
OVERLAP_SD = 11.40520671261509
OVERLAP_SIGMA = 3.0

# Analytic expectation 3 -- TOP-1 HITS over m queries: mean m/n = 1.0,
# binomial sd sqrt(m*(1/n)*(1-1/n)) = 0.9960860906568267. With a mean of
# one hit this band is weak; it is reported for completeness, not power.
TOP1_EXPECTED = 1.0
TOP1_SD = 0.9960860906568267
TOP1_SIGMA = 3.0

# ---------------------------------------------------------------------------
# Malformed-branch diagnostic pin
# ---------------------------------------------------------------------------
# The malformed branch (commit MALFORMED_SOURCE_COMMIT) returns
# np.zeros(7, dtype=float32) from embed_text -- a 1-D vector. The adapter's
# refusal note for that exact violation, with 7 captions, is pinned here so
# the diagnostic is itself a contract.

MALFORMED_CAPTION_COUNT = 7
MALFORMED_NOTE_CONTAINS = [
    "embed_text returned shape (7,)",
    "expected (7, 64)",
    "ndim=1",
    "dtype=float32",
    "width=7",
    "run refused before scoring",
    "per-caption retry cannot rescue",
    "width 7 is below the fixed embedding width 64",
]
