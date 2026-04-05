"""Smoke import tests for core modules."""


def test_core_imports() -> None:
    import api.main  # noqa: F401
    import src.model.recommend  # noqa: F401
    import src.utils.profile  # noqa: F401
