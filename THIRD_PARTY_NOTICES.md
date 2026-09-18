# Third-party notices

Pokefly's optic-column parser and partner-based receptor placement adapt
`flybrain/build.py`. The live brain viewer and local event-stream design adapt
`sshfighter/dashboard.html` and `sshfighter/fly_dashboard.py` from
[alextitonis/fly.ai](https://github.com/alextitonis/fly.ai), pinned to revision
`9af6fb6345cb55e359e71d09fae99f2c6dd46456`. Pokefly changes the scene display,
telemetry, input projection, controls, and validation reporting. The viewer is
not a reconstruction of neuronal morphology or a synaptic transmission movie.

## fly.ai — MIT License

Copyright (c) 2026 alextitonis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Connectome and column annotations

The MaleCNS v1.0 connectome is work of FlyEM (HHMI Janelia), the University of
Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Dataset
licensing and attribution are separate from the MIT application code:
[MaleCNS downloads, CC BY 4.0](https://male-cns.janelia.org/download/).

The optic-column table is sourced from
[flyconnectome/2025malecns](https://github.com/flyconnectome/2025malecns), revision
`67767d2233657983993ff6c2be48e836a935863c`, file
`supplemental_data/optic-column-type-assignments-v1.0.xlsx`, SHA-256
`d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3`.
Derived retina data retain annotation provenance. Its normalization into screen
coordinates is a Pokefly engineering approximation, not measured fly optics.

## Motor identity and read-only game instrumentation

The MN9/CB0701 mapping uses curated Virtual Fly Brain annotations for
[MaleCNS 10331](https://www.virtualflybrain.org/blog/2022/01/01/mn9_l-malecns10331-vfb_jrmc20e1/)
and [MaleCNS 16949](https://www.virtualflybrain.org/blog/2022/01/01/mn9_r-malecns16949-vfb_jrmc20e2/),
cross-checked by body ID and alias against the downloaded brain. Experimental
proboscis motor functions are described in [McKellar et al.](https://elifesciences.org/articles/54978).

Reward instruction addresses and RAM definitions are factual references from
[pret/pokered](https://github.com/pret/pokered): symbols revision
`3f618d59edf43918f48f5e558c34e04cb2fc5619`, source revision
`a1a22aaf84d1675bcdbaeb194592379d586d838e`. Pokefly does not package the game,
its disassembled source, or assembled game code. Short instruction signatures
only verify observer hook locations in the supported user-supplied ROM.

The ROM is supplied by the user and is not distributed with this project.
