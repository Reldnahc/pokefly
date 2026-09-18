"""Pixel-to-photoreceptor input, with a documented, fixed 2-D column chart.

No game state, object detector, OCR, or learned encoder is used here. This is an
approximate *screen mapping*, not a reconstruction of a fly's optical geometry.
Optic-column parsing and strongest-partner assignment adapt fly.ai/build.py
(MIT, alextitonis); see THIRD_PARTY_NOTICES.md. Unplaced cells receive zero drive.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAPPING_VERSION = "split-eyes-axial-2d-v1"
COLUMNS_REVISION = "67767d2233657983993ff6c2be48e836a935863c"
COLUMNS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/2025malecns/"
    f"{COLUMNS_REVISION}/supplemental_data/optic-column-type-assignments-v1.0.xlsx"
)
COLUMNS_SHA256 = "d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def population_signature(ids: np.ndarray, visual: np.ndarray) -> str:
    digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes())
    digest.update(np.asarray(visual, dtype="<i8").tobytes())
    return digest.hexdigest()


def fetch_columns(data: Path) -> Path:
    """Download one small, public annotation file; never overwrite an existing file."""
    target = data / "raw" / "optic-columns.xlsx"
    if target.exists():
        if file_sha256(target) != COLUMNS_SHA256:
            raise ValueError(f"Optic-column checksum mismatch: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"optic-columns-{uuid.uuid4().hex}.part")
    try:
        with urllib.request.urlopen(COLUMNS_URL, timeout=30) as response:
            with partial.open("xb") as output:
                while chunk := response.read(64 * 1024):
                    output.write(chunk)
        if file_sha256(partial) != COLUMNS_SHA256:
            raise ValueError("Downloaded optic-column annotations failed SHA-256 verification")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def optic_columns(path: Path) -> dict[int, tuple[str, int, int]]:
    """Read L1/R7/R8 body IDs and both hex-column coordinates without Excel dependencies."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    result: dict[int, tuple[str, int, int]] = {}
    with zipfile.ZipFile(path) as archive:
        strings = [
            "".join(element.itertext())
            for element in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        for sheet in (1, 2):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
            for row in root.findall("s:sheetData/s:row", ns)[1:]:
                cells = {}
                for cell in row:
                    value = cell.find("s:v", ns)
                    if value is not None and value.text is not None:
                        cells[re.sub(r"\d", "", cell.attrib["r"])] = (
                            strings[int(value.text)] if cell.get("t") == "s" else value.text
                        )
                match = re.fullmatch(r"ME_([LR])_col_(\d+)_(\d+)", cells.get("A", ""))
                if not match:
                    continue
                side, h1, h2 = match.groups()
                location = (side, int(h1), int(h2))
                for name in ("B", "C", "E"):
                    try:
                        body = int(cells.get(name, -1))
                    except ValueError:
                        continue
                    if body > 0:
                        if body in result and result[body] != location:
                            raise ValueError(f"Conflicting optic columns for body {body}")
                        result[body] = location
    if not result:
        raise ValueError("No optic-column assignments found")
    return result


@dataclass
class RetinaMap:
    indices: np.ndarray
    body_ids: np.ndarray
    u: np.ndarray
    v: np.ndarray
    provenance: np.ndarray  # 0 = unplaced, 1 = directly annotated, 2 = strongest partner
    metadata: dict

    def __post_init__(self) -> None:
        length = len(self.indices)
        if not length or any(
            array.shape != (length,)
            for array in (self.indices, self.body_ids, self.u, self.v, self.provenance)
        ):
            raise ValueError("Retina arrays must be nonempty, equally sized 1-D arrays")
        if self.indices.dtype.kind not in "iu" or np.any(self.indices < 0):
            raise ValueError("Retina indices must be nonnegative integers")
        if len(np.unique(self.indices)) != length:
            raise ValueError("Duplicate retina neuron indices")
        if not np.isin(self.provenance, [0, 1, 2]).all() or not self.mapped.any():
            raise ValueError("Retina must contain mapped cells with valid provenance")
        for coordinate in (self.u[self.mapped], self.v[self.mapped]):
            if not np.isfinite(coordinate).all() or np.any((coordinate < 0) | (coordinate > 1)):
                raise ValueError("Mapped retinal coordinates must be finite and in [0, 1]")
        if self.metadata.get("mapping") != MAPPING_VERSION:
            raise ValueError("Unsupported retina mapping version; run 'pokefly vision-prepare'")

    @property
    def mapped(self) -> np.ndarray:
        return self.provenance > 0

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """RGB mean luminance and bilinear sampling. All constants stay fixed."""
        if (
            frame.ndim != 3
            or frame.shape[2] not in (3, 4)
            or min(frame.shape[:2]) < 2
            or frame.dtype != np.uint8
        ):
            raise ValueError("Expected an HxWx3/4 uint8 screen with H,W >= 2")
        gray = frame[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
        height, width = gray.shape
        valid = self.mapped
        x, y = self.u[valid] * (width - 1), self.v[valid] * (height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        dx, dy = x - x0, y - y0
        drive = np.zeros(len(self.indices), np.float32)
        drive[valid] = (
            gray[y0, x0] * (1 - dx) * (1 - dy)
            + gray[y0, x1] * dx * (1 - dy)
            + gray[y1, x0] * (1 - dx) * dy
            + gray[y1, x1] * dx * dy
        )
        return np.clip(drive, 0, 1)

    def summary(self) -> dict:
        return {
            **self.metadata,
            "photoreceptors": len(self.indices),
            "direct": int(np.count_nonzero(self.provenance == 1)),
            "inferred": int(np.count_nonzero(self.provenance == 2)),
            "unplaced_zero_driven": int(np.count_nonzero(self.provenance == 0)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            indices=self.indices,
            body_ids=self.body_ids,
            u=self.u,
            v=self.v,
            provenance=self.provenance,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path, brain_data: Path) -> RetinaMap:
        with np.load(path, allow_pickle=False) as data:
            retina = cls(
                **{name: data[name] for name in ("indices", "body_ids", "u", "v", "provenance")},
                metadata=json.loads(str(data["metadata"])),
            )
        with np.load(brain_data, allow_pickle=False) as brain:
            signature = population_signature(brain["ids"], brain["visual"])
            if (
                retina.metadata.get("population_signature") != signature
                or not np.array_equal(retina.indices, brain["visual"])
                or not np.array_equal(retina.body_ids, brain["ids"][brain["visual"]])
            ):
                raise ValueError("Retina belongs to different brain data; rebuild it")
        return retina


def build_retina(data: Path) -> RetinaMap:
    """Keep both column coordinates; infer R1-6 placement via annotated partners.

    Each eye covers half the screen, both cover its full height. h1 and h2 are
    normalized independently (an axial-grid chart, not measured visual angles).
    The left chart is mirrored horizontally, as in the upstream azimuth map.
    """
    from scipy import sparse

    columns_path = fetch_columns(data)
    columns = optic_columns(columns_path)
    with np.load(data / "brain.npz", allow_pickle=False) as meta:
        ids, visual, sides = meta["ids"], meta["visual"], meta["side"].astype(str)
    weights = sparse.load_npz(data / "weights.npz").tocsr()
    if weights.shape != (len(ids), len(ids)):
        raise ValueError("Brain IDs and connectivity dimensions do not match")
    known = np.array([i for i, body in enumerate(ids) if int(body) in columns], dtype=np.int64)
    if not len(known):
        raise ValueError("Optic-column annotations do not match this brain")
    links = abs(weights[known][:, visual]).tocsc()
    u = np.full(len(visual), np.nan, np.float32)
    v = np.full_like(u, np.nan)
    provenance = np.zeros(len(visual), np.uint8)
    extents = {}
    for side in "LR":
        grid = np.array([(h1, h2) for s, h1, h2 in columns.values() if s == side])
        extents[side] = (grid.min(axis=0), np.maximum(grid.max(axis=0) - grid.min(axis=0), 1))
    for k, neuron in enumerate(visual):
        location = columns.get(int(ids[neuron]))
        source = 1
        if location is None:
            start, end = links.indptr[k : k + 2]
            candidates = [
                edge
                for edge in range(start, end)
                if sides[neuron] not in ("L", "R")
                or columns[int(ids[known[links.indices[edge]]])][0] == sides[neuron]
            ]
            if candidates:
                best = max(candidates, key=lambda edge: links.data[edge])
                location = columns[int(ids[known[links.indices[best]]])]
                source = 2
        if location is None:
            continue
        side, h1, h2 = location
        low, span = extents[side]
        horizontal, vertical = (np.array([h1, h2]) - low) / span
        u[k] = 0.5 * (1 - horizontal) if side == "L" else 0.5 + 0.5 * horizontal
        v[k] = vertical
        provenance[k] = source
    retina = RetinaMap(
        visual,
        ids[visual],
        u,
        v,
        provenance,
        {
            "mapping": MAPPING_VERSION,
            "population_signature": population_signature(ids, visual),
            "source_url": COLUMNS_URL,
            "source_sha256": file_sha256(columns_path),
            "weights_sha256": file_sha256(data / "weights.npz"),
            "projection": "approximate split-eye normalized axial chart; not measured optics",
            "transduction": "RGB mean / 255, bilinear sampling; unplaced cells = 0",
            "biological_vision_validated": False,
        },
    )
    retina.save(data / "retina.npz")
    return retina
