import numpy as np

from pokefly.brain import image_features


def test_visual_features_preserve_side_and_only_report_actual_motion() -> None:
    frame = np.full((144, 160, 4), 255, np.uint8)
    frame[:, :80, :3] = 0
    first = image_features(frame, None)
    assert first[0] == 1 and first[4] == 0  # Darkness belongs on the left.
    assert first[2] == first[6] == 0  # No fabricated motion on first observation.
    previous = frame[:, :, :3].astype(np.float32).mean(axis=2) / 255
    np.testing.assert_array_equal(image_features(frame, previous), first)
    changed = frame.copy()
    changed[:, :80, :3] = 255
    motion = image_features(changed, previous)
    assert motion[2] == 1 and motion[6] == 0
    assert np.isfinite(motion).all() and (motion >= 0).all() and (motion <= 1).all()
