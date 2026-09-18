"""Fetch pinned, licensed neural-calibration DATA; never execute upstream code.

Reference model: Abijah Kajabika, fruit-fly-brain-research; MaleCNS CC BY 4.0
and flyvis/MIT. See THIRD_PARTY_NOTICES.md and the downloaded data license.
No ROM, game labels, motor adapter, or policy is downloaded.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

from pokefly.runner import write_json

REVISION = "c28066a5b9eff03efdb002f6779980d29f70634c"
ROOT = "https://raw.githubusercontent.com/AbijahKaj/fruit-fly-brain-research/" + REVISION
ASSETS = {
    "app/public/graphs/optic-v2.json": (29054, "7a9622812a7ac7798fd3ce5573d5ec73a303ba4b"),
    "app/public/graphs/optic-v2.bin": (24886492, "4fc056cc688c90a59604815d76967c46ea2f7304"),
    "app/public/graphs/fitted-params.json": (163970, "f0324f4c3363b8099cd010d96a9f39028f75131c"),
    "app/public/graphs/flyvis-params.json": (66564, "2dfc557cccbe8cd930812aa52739dc743832d36f"),
    "DATA-LICENSE.md": (1476, "d254ba19551b9483dd9ea7aef4fdb42bc4f11a24"),
    "LICENSE": (1072, "9e7cc5d1fdaaddbae75b511772438bf032ee36c1"),
}


def validate_asset(data, size, git_sha):
    if len(data) != size:
        raise ValueError("Pinned visual reference size mismatch")
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != git_sha:
        raise ValueError("Pinned visual reference Git-blob hash mismatch")


def main():
    output = Path("fly-data") / ("visual-reference-" + REVISION[:8])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": ROOT, "revision": REVISION, "assets": {}}
    for name, (size, expected) in ASSETS.items():
        destination = output / Path(name).name
        if destination.exists():
            data = destination.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(ROOT + "/" + name, timeout=30) as response:
                        data = response.read(size + 1)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            validate_asset(data, size, expected)
            with destination.open("xb") as stream:
                stream.write(data)
        validate_asset(data, size, expected)
        manifest["assets"][destination.name] = {
            "url": ROOT + "/" + name,
            "bytes": size,
            "git_blob_sha1": expected,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        print("Verified data", destination, flush=True)
    write_json(output / "manifest.json", manifest)
    print("Reference:", output, flush=True)


if __name__ == "__main__":
    main()
