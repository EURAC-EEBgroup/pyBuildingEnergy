"""Packaging regressions for runtime resources and project metadata."""

from pathlib import Path

from pybuildingenergy.source.functions import get_buildings_demos


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_archetypes_are_loadable():
    archetypes = get_buildings_demos()
    assert archetypes is not None


def test_packaging_uses_repository_filename_casing():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "Readme.md"' in pyproject
    assert (ROOT / "Readme.md").is_file()


def test_manifest_excludes_generated_bytecode():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "global-exclude *.py[cod]" in manifest
    assert "recursive-include tests *.py" in manifest
