import pytest

from src.core.document_parser import validate_ocr_dpi


@pytest.mark.parametrize("value", [250.5, 199.1, float("nan"), float("inf")])
def test_fractional_or_non_finite_dpi_is_rejected(value):
    with pytest.raises(ValueError, match="integer between 150 and 400"):
        validate_ocr_dpi(value)


@pytest.mark.parametrize("value, expected", [(250.0, 250), ("300", 300), (400, 400)])
def test_integral_dpi_values_are_accepted(value, expected):
    assert validate_ocr_dpi(value) == expected
