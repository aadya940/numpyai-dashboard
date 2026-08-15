"""Tests for pure helpers that don't need an LLM."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from numpyai_dashboard._utils import (
    NumpyMetadataCollector,
    _SubmodulePackage,
    clean_code,
    optional_globals,
)
from numpyai_dashboard._validator import NumpyValidator


def test_clean_code_strips_fences():
    src = "```python\nprint('hi')\n```"
    assert clean_code(src) == "print('hi')"


def test_clean_code_no_fence_is_noop():
    assert clean_code("x = 1") == "x = 1"


def test_metadata_1d():
    arr = np.array([1.0, 2.0, np.nan])
    md = NumpyMetadataCollector().metadata(arr)
    assert md["shape"] == (3,)
    assert md["dims"] == 1
    assert md["has_nan"] is True


def test_metadata_scalar_zero_dim_does_not_crash():
    arr = np.array(5)
    md = NumpyMetadataCollector().metadata(arr)
    assert md["dims"] == 0
    assert md["shape"] == ()


def test_output_metadata_ndarray():
    md = NumpyMetadataCollector.collect_output_metadata(np.array([1, 2, 3]))
    assert md["type"] == "ndarray"
    assert md["shape"] == (3,)
    assert md["first_elements"] == [1, 2, 3]


def test_output_metadata_none():
    assert NumpyMetadataCollector.collect_output_metadata(None)["is_none"] is True


def test_validator_accepts_good_code():
    assert NumpyValidator().validate_code("output = 1 + 2") is True


def test_validator_rejects_bad_code():
    assert NumpyValidator().validate_code("output = 1 +") is False


# --------------------------------------------------------------------------
# optional libraries offered to generated code
# --------------------------------------------------------------------------


def test_optional_globals_never_raises():
    """Every library is optional; missing ones are absent, not an error."""
    assert isinstance(optional_globals(), dict)


def test_optional_globals_only_exposes_installed_libraries():
    ns = optional_globals()
    for name, module in [
        ("plt", "matplotlib"),
        ("sklearn", "sklearn"),
        ("scipy", "scipy"),
    ]:
        installed = importlib.util.find_spec(module) is not None
        assert (
            name in ns
        ) is installed, f"{name} present={name in ns}, installed={installed}"


def test_scipy_submodules_resolve_when_present():
    """A bare `import scipy` leaves scipy.stats unresolved; the proxy fixes it."""
    scipy = optional_globals().get("scipy")
    if scipy is None:
        pytest.skip("scipy not installed")
    for sub in ("stats", "optimize", "signal", "linalg", "interpolate"):
        assert hasattr(scipy, sub), f"scipy.{sub} would fail in generated code"


def test_sklearn_submodules_resolve_when_present():
    sklearn = optional_globals().get("sklearn")
    if sklearn is None:
        pytest.skip("scikit-learn not installed")
    for sub in ("linear_model", "preprocessing", "cluster", "metrics"):
        assert hasattr(sklearn, sub), f"sklearn.{sub} would fail in generated code"


# The proxy itself is exercised against stdlib packages that also defer their
# submodule imports, so these run everywhere rather than only where scipy is.


def test_proxy_imports_submodule_on_access():
    import xml

    assert not hasattr(xml, "etree") or True  # bare package may lack it
    assert _SubmodulePackage(xml).etree.__name__ == "xml.etree"


def test_proxy_reaches_nested_attributes():
    import xml

    assert _SubmodulePackage(xml).etree.ElementTree.__name__ == "xml.etree.ElementTree"


def test_proxy_passes_through_real_attributes():
    import xml

    assert _SubmodulePackage(xml).__name__ == "xml"


def test_proxy_raises_attribute_error_not_import_error():
    """Generated code sees a normal AttributeError, not a leaked ImportError."""
    import xml

    proxy = _SubmodulePackage(xml)
    with pytest.raises(AttributeError, match="no attribute or submodule"):
        _ = proxy.nope
