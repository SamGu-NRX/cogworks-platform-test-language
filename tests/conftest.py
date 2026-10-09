import os
import sys

# Make the shared adapter and fixture modules importable from test files
# regardless of where pytest is invoked from. This only adds THIS tests/
# directory; it does not put the repository root on sys.path, and it never
# imports the submission module itself (tests load it via local_adapter).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
