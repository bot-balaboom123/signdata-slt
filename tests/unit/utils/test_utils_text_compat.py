"""Compatibility tests for ``signdata.utils.text``."""

from signdata.datasets._ingestion.text import TextProcessingConfig, normalize_text


def test_normalize_text_compat_import():
    assert normalize_text("  CafÃ©  \n") == "Café"


def test_text_processing_config_compat_import():
    cfg = TextProcessingConfig()
    assert cfg.fix_encoding is True
    assert cfg.normalize_whitespace is True
