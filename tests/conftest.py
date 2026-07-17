"""
conftest.py
-----------
Global pytest fixtures and path configuration for the semantic plagiarism
detector test suite.

Path Bootstrap
~~~~~~~~~~~~~~
Inserts the repository root into sys.path so that `src`, `app`, and `utils`
packages are importable when running `pytest` directly from the project root.

This acts as a robust fallback guarantee alongside the `pythonpath = .`
directive in pytest.ini, ensuring compatibility with older pytest versions
(< 7.0) that do not support the pythonpath ini option.

Sentence Transformers Stub
~~~~~~~~~~~~~~~~~~~~~~~~~~
Stubs out sentence_transformers so tests can run without a fully compatible
TensorFlow / Keras installation. The embedding_model tests mock _get_model()
directly, so no real model is loaded.
"""
import sys
import types
import shutil
import pathlib
import pytest
from unittest.mock import MagicMock

# ── Repository Root Path Bootstrap ────────────────────────────────────────────
# Resolve the repository root (two levels up from this conftest.py file) and
# prepend it to sys.path so `import src.*`, `import app.*`, `import utils.*`
# all resolve correctly regardless of how pytest is invoked.
_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Sentence Transformers Stub ────────────────────────────────────────────────
# Stub sentence_transformers before any test module imports it.
if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = MagicMock  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub

# ── Tesseract OCR Availability ────────────────────────────────────────────────
# Detect whether the Tesseract binary is available on PATH.
# Tests decorated with @pytest.mark.skipif(not TESSERACT_AVAILABLE, ...)
# will be gracefully skipped on machines that don't have Tesseract installed
# (e.g. local developer machines, basic CI environments).
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
