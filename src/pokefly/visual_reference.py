"""Pinned neural-parameter reference data, never an external game policy.

Graph format/parameter names originate in Abijah Kajabika's MIT-licensed
fruit-fly-brain-research; graph data are CC BY 4.0. See THIRD_PARTY_NOTICES.md.
Only typed arrays and JSON numbers are loaded; no pickle or upstream code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pokefly.checkpoint import sha256

REVISION = "c28066a5b9eff03efdb002f6779980d29f70634c"
ALIASES = {"TmY9a": "TmY9", "TmY9b": "TmY9"}
DTYPES = {
    "int8": np.dtype("i1"), "int16": np.dtype("<i2"), "int32": np.dtype("<i4"),
    "float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
}


def typed_array(blob, spec):
    dtype = DTYPES.get(spec["dtype"])
    offset, length = spec["offset"], spec["length"]
    if (
        dtype is None or not isinstance(offset, int) or not isinstance(length, int)
        or min(offset, length) < 0 or offset % dtype.itemsize
        or offset + length * dtype.itemsize > len(blob)
    ):
        raise ValueError("Invalid visual-reference typed-array bounds")
    return np.frombuffer(blob, dtype=dtype, count=length, offset=offset)


class VisualReference:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["revision"] != REVISION:
            raise ValueError("Unrecognized visual-reference revision")
        for name in ("optic-v2.json", "optic-v2.bin", "fitted-params.json", "flyvis-params.json"):
            if sha256(directory / name) != self.manifest["assets"][name]["sha256"]:
                raise ValueError("Visual-reference file hash mismatch")
        self.header = json.loads((directory / "optic-v2.json").read_text())
        self.parameters = json.loads((directory / "fitted-params.json").read_text())
        self.original_parameters = json.loads((directory / "flyvis-params.json").read_text())
        self.blob = (directory / "optic-v2.bin").read_bytes()
        self.arrays = {
            name: typed_array(self.blob, spec) for name, spec in self.header["arrays"].items()
        }
        self.n, self.m = self.header["units"]["count"], self.header["edges"]["count"]
        if self.header["version"] != 2 or not 0 < self.n <= 166700 or not 0 < self.m < 30_000_000:
            raise ValueError("Unexpected visual-reference graph size or version")
        for name, array in self.arrays.items():
            expected = self.n if name.startswith("units.") else self.m
            if not name.startswith("columns.") and len(array) != expected:
                raise ValueError("Visual-reference graph arrays do not align")
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite visual-reference data")
        bodies = self.arrays["units.bodyId"]
        if (bodies <= 0).any() or (bodies != bodies.astype(np.int64)).any():
            raise ValueError("Reference contains non-anatomical body identifiers")
        self.body_ids = bodies.astype(np.int64)
        if len(np.unique(self.body_ids)) != self.n:
            raise ValueError("Duplicate reference body identifiers")
        for field, maximum in (
            ("units.type", len(self.header["types"])), ("units.side", len(self.header["sides"])),
            ("units.role", len(self.header["roles"])),
            ("edges.pre", self.n), ("edges.post", self.n),
        ):
            values = self.arrays[field]
            if values.dtype.kind not in "iu" or (values < 0).any() or (values >= maximum).any():
                raise ValueError("Reference graph index out of bounds")
        self.types = np.array([r["name"] for r in self.header["types"]])[
            self.arrays["units.type"]
        ]
        self.sides = np.array(self.header["sides"])[self.arrays["units.side"]]
        self.roles = np.array(self.header["roles"])[self.arrays["units.role"]]
        self.canonical_types = np.array([ALIASES.get(str(t), str(t)) for t in self.types])

    def match_body_ids(self, body_ids):
        ordered = np.argsort(body_ids, kind="stable")
        where = np.searchsorted(body_ids[ordered], self.body_ids)
        if (where >= len(body_ids)).any() or not np.array_equal(
            body_ids[ordered[where]], self.body_ids
        ):
            raise ValueError("Reference neurons are missing from the original fly")
        return ordered[where]
