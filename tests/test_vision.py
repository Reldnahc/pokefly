import zipfile

import numpy as np
import pytest

from pokefly.vision import MAPPING_VERSION, RetinaMap, optic_columns, population_signature


@pytest.fixture
def retina():
    return RetinaMap(
        np.arange(6),
        np.arange(100, 106),
        np.array([0, 1, 0, 1, 0.5, np.nan]),
        np.array([0, 0, 1, 1, 0.5, np.nan]),
        np.array([1, 1, 2, 2, 1, 0]),
        {"mapping": MAPPING_VERSION},
    )


def test_spatial_mapping_retains_both_axes(retina):
    left = np.zeros((2, 2, 3), np.uint8)
    left[:, 0] = 255
    top = np.zeros_like(left)
    top[0] = 255
    np.testing.assert_allclose(retina.encode(left), [1, 0, 1, 0, 0.5, 0])
    np.testing.assert_allclose(retina.encode(top), [1, 1, 0, 0, 0.5, 0])


def test_unplaced_cells_are_silent_not_assigned_fake_coordinates(retina):
    np.testing.assert_equal(retina.encode(np.full((2, 2, 4), 255, np.uint8)), [1, 1, 1, 1, 1, 0])
    assert retina.summary()["unplaced_zero_driven"] == 1


def test_grayscale_bilinear_and_alpha_handling(retina):
    frame = np.zeros((2, 2, 4), np.uint8)
    frame[0, 0, :3] = [255, 0, 0]
    frame[:, :, 3] = 255
    drive = retina.encode(frame)
    np.testing.assert_allclose(drive, [1 / 3, 0, 0, 0, 1 / 12, 0], atol=1e-7)
    assert drive.dtype == np.float32


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2), np.uint8),
        np.zeros((2, 2, 2), np.uint8),
        np.zeros((1, 2, 3), np.uint8),
        np.zeros((2, 2, 3), np.float32),
    ],
)
def test_invalid_frame_rejected(retina, frame):
    with pytest.raises(ValueError, match="screen"):
        retina.encode(frame)


def test_mapping_round_trip_and_population_guard(retina, tmp_path):
    ids = retina.body_ids
    retina.metadata["population_signature"] = population_signature(ids, retina.indices)
    retina.save(tmp_path / "retina.npz")
    np.savez(tmp_path / "brain.npz", ids=ids, visual=retina.indices)
    restored = RetinaMap.load(tmp_path / "retina.npz", tmp_path / "brain.npz")
    np.testing.assert_equal(restored.u, retina.u)
    np.savez(tmp_path / "brain.npz", ids=ids + 1, visual=retina.indices)
    with pytest.raises(ValueError, match="different brain"):
        RetinaMap.load(tmp_path / "retina.npz", tmp_path / "brain.npz")


def test_coordinate_validation():
    with pytest.raises(ValueError, match="coordinates"):
        RetinaMap(
            np.array([0]),
            np.array([1]),
            np.array([2.0]),
            np.array([0.5]),
            np.array([1]),
            {"mapping": MAPPING_VERSION},
        )


def test_column_parser_uses_both_coordinates(tmp_path):
    path = tmp_path / "columns.xlsx"
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f"<sst {ns}><si><t>ME_L_col_12_34</t></si><si><t>ME_R_col_3_9</t></si></sst>",
        )
        for sheet in (1, 2):
            archive.writestr(
                f"xl/worksheets/sheet{sheet}.xml",
                f"<worksheet {ns}><sheetData>"
                '<row r="1"/><row r="2">'
                f'<c r="A2" t="s"><v>{sheet - 1}</v></c>'
                f'<c r="B2"><v>{sheet * 10}</v></c>'
                f'<c r="C2"><v>{sheet * 10 + 1}</v></c>'
                f'<c r="E2"><v>{sheet * 10 + 2}</v></c>'
                "</row></sheetData></worksheet>",
            )
    columns = optic_columns(path)
    assert columns[10] == columns[11] == columns[12] == ("L", 12, 34)
    assert columns[20] == columns[21] == columns[22] == ("R", 3, 9)
