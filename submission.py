"""A deliberately weak language-search submission for testing the CogWorks platform.

This is not a capstone solution. It exists so the hosted path (fetch, prepare,
evaluate, score, report) can be exercised on code nobody needs to keep private.

Text is embedded by hashing each lowercase word into one of 64 buckets and
counting, so captions that share words land near each other. Images go through
a fixed random projection of their 512-d descriptors, which has no relation to
the text buckets, so text-to-image search should sit close to chance.
Everything is deterministic: CRC-32 rather than Python's salted `hash`, and a
seeded generator for the projection.
"""

import re
import zlib

import numpy as np

DIMENSIONS = 64
WORD = re.compile(r"[a-z0-9]+")


def _hashed_words(text):
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    for word in WORD.findall(text.lower()):
        vector[zlib.crc32(word.encode("utf-8")) % DIMENSIONS] += 1.0
    return vector


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashedWordSearch:
    def __init__(self):
        generator = np.random.RandomState(20261002)
        self.projection = generator.standard_normal((512, DIMENSIONS)).astype(np.float32)
        self.image_ids = []
        self.image_vectors = np.zeros((0, DIMENSIONS), dtype=np.float32)

    def embed_text(self, captions):
        rows = [_hashed_words(caption) for caption in captions]
        if not rows:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        return np.stack(rows)

    def embed_images(self, descriptors):
        return np.asarray(descriptors, dtype=np.float32) @ self.projection

    def prepare_database(self, image_ids, descriptors):
        self.image_ids = [int(image_id) for image_id in image_ids]
        self.image_vectors = _unit_rows(self.embed_images(descriptors))

    def search(self, query, k):
        query_vector = _hashed_words(query)
        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm
        scores = self.image_vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[: int(k)]
        return [self.image_ids[index] for index in order]


def create_submission(resources):
    return HashedWordSearch()
