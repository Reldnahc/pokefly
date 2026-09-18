import importlib.util
from pathlib import Path

import pytest


def test_menu_timer_keeps_unknown_prefix_and_includes_nested_menu_time():
    script = Path(__file__).parents[1] / "scripts" / "audit_menu_time.py"
    spec = importlib.util.spec_from_file_location("menu_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = module.MenuClock()
    clock.event("overworld", 10)
    clock.event("menu", 20)
    clock.advance(50)
    clock.event("overworld", 80)
    clock.advance(100)
    assert dict(clock.frames) == {"unknown": 10, "not_start_menu": 30, "menu": 60}
    assert clock.entries == 1
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(99)
