"""Fixed button channels shared by the decoder, emulator, and measurements."""

from collections import Counter
from collections.abc import Mapping

DIRECTIONS = ("up", "down", "left", "right")
FUNCTION_BUTTONS = ("a", "b", "start")
BUTTONS = DIRECTIONS + FUNCTION_BUTTONS
# Manual/legacy policy choices remain single-button commands.
ACTIONS = BUTTONS + ("wait",)


def pressed_buttons(action: str) -> tuple[str, ...]:
    """Parse a canonical command: Wait, one button, or direction+function.

    No opposite directions, diagonals, A+B, duplicates, or implicit reordering.
    Validation happens before any emulator input is queued.
    """
    if action == "wait":
        return ()
    if isinstance(action, str):
        parts = tuple(action.split("+"))
        if len(parts) == 1 and parts[0] in BUTTONS:
            return parts
        if len(parts) == 2 and parts[0] in DIRECTIONS and parts[1] in FUNCTION_BUTTONS:
            return parts
    raise ValueError(f"Invalid button command: {action!r}")


def count_buttons(actions: Mapping[str, int]) -> dict[str, int]:
    """Count delivered buttons, distinct from the number of decision windows."""
    counts = Counter({button: 0 for button in BUTTONS})
    for action, count in actions.items():
        for button in pressed_buttons(action):
            counts[button] += count
    return dict(counts)
