from __future__ import annotations

import hashlib
from pathlib import Path

RED_ENGLISH_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"


class RomError(ValueError):
    """Raised when a usable, safely ignored ROM cannot be identified."""


def discover_rom(root: Path) -> Path:
    """Look in the project root and roms/, never installed packages or caches."""
    candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".gb"]
    if (root / "roms").is_dir():
        candidates.extend(
            p for p in (root / "roms").rglob("*") if p.is_file() and p.suffix.lower() == ".gb"
        )
    candidates.sort()
    if not candidates:
        raise RomError(f"No .gb ROM found below {root}")
    if len(candidates) > 1:
        names = ", ".join(str(path.relative_to(root)) for path in candidates)
        raise RomError(f"More than one .gb ROM found; pass --rom explicitly: {names}")
    return candidates[0].resolve()


def resolve_rom(value: str | None, root: Path) -> Path:
    path = Path(value).expanduser() if value else discover_rom(root)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RomError(f"ROM does not exist: {path}")
    if path.suffix.lower() != ".gb":
        raise RomError("Pokemon Red must be supplied as a .gb cartridge dump")
    return path


def validate_rom(path: Path) -> str:
    """Identify the supported English release before trusting its WRAM layout."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha1").hexdigest()
    if digest != RED_ENGLISH_SHA1:
        raise RomError(
            "This ROM differs from the supported English Red (UE) release "
            f"(SHA-1 {digest}). Its memory layout needs a separate adapter."
        )
    return digest
