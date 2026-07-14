from pathlib import Path, PurePosixPath

from workspace_paths import configured_path, is_absolute_path, resolve_workspace_path


def test_windows_drive_path_is_absolute_on_every_host_platform() -> None:
    assert is_absolute_path("C:/model/outputs/high_return_v2")
    assert is_absolute_path(r"D:\codex\daily-market-data")


def test_posix_absolute_path_is_absolute() -> None:
    assert is_absolute_path(PurePosixPath("/srv/quant/data"))


def test_relative_path_is_resolved_under_project_root() -> None:
    root = Path("project-root")
    assert resolve_workspace_path(root, "outputs/daily") == root / "outputs/daily"


def test_windows_absolute_path_is_not_prefixed_by_posix_style_root() -> None:
    resolved = resolve_workspace_path(
        Path("C:/model"),
        PurePosixPath("C:/model/outputs/high_return_v2"),
    )
    assert str(resolved).replace("\\", "/") == "C:/model/outputs/high_return_v2"


def test_configured_path_prefers_environment(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_TEST_ROOT", "/srv/quant/data")
    assert configured_path("QUANT_TEST_ROOT", "external_data") == Path("/srv/quant/data")


def test_configured_path_uses_relative_portable_default(monkeypatch) -> None:
    monkeypatch.delenv("QUANT_TEST_ROOT", raising=False)
    assert configured_path("QUANT_TEST_ROOT", "external_data") == Path("external_data")
