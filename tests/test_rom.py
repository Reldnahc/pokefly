from pathlib import Path

import pytest

from pokefly.rom import RomError, discover_rom, resolve_rom, validate_rom


def test_discover_single_rom(tmp_path: Path) -> None:
    rom = tmp_path / "PokemonRed.gb"
    rom.write_bytes(b"test")
    assert discover_rom(tmp_path) == rom.resolve()


def test_discover_rejects_multiple_roms(tmp_path: Path) -> None:
    (tmp_path / "red.gb").write_bytes(b"test")
    (tmp_path / "blue.gb").write_bytes(b"test")
    with pytest.raises(RomError, match="More than one"):
        discover_rom(tmp_path)


def test_resolve_rejects_non_rom(tmp_path: Path) -> None:
    path = tmp_path / "not-a-rom.txt"
    path.write_text("test")
    with pytest.raises(RomError, match=".gb"):
        resolve_rom(str(path), tmp_path)


def test_discovery_excludes_installed_emulator_roms(tmp_path: Path) -> None:
    installed = tmp_path / ".venv" / "pyboy"
    installed.mkdir(parents=True)
    (installed / "default_rom.gb").write_bytes(b"test")
    actual = tmp_path / "Pokemon Red.GB"
    actual.write_bytes(b"test")
    assert discover_rom(tmp_path) == actual.resolve()


def test_unsupported_rom_rejected_before_using_memory_layout(tmp_path: Path) -> None:
    rom = tmp_path / "wrong.gb"
    rom.write_bytes(b"not the supported release")
    with pytest.raises(RomError, match="supported English Red"):
        validate_rom(rom)
