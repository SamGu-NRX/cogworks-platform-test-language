# CogWorks platform test submission

A deliberately weak Week 3 language-search submission, used to test that the
CogWorks portal can fetch, prepare, evaluate and score a repository end to end.
It is not a capstone solution and is not meant to score well.

`submission.py` embeds text by hashing words into 64 buckets and embeds images
with a fixed random projection, so text-to-image search should land near chance.

Branches:

- `main`: a valid submission.
- `malformed`: `embed_text` returns a 1-D vector instead of a `(rows, D)` matrix,
  so the benchmark should report the shape problem as a note on the run.
