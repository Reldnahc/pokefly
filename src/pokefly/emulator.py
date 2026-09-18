from __future__ import annotations

import io
from pathlib import Path

from pokefly.actions import ACTIONS as ACTIONS
from pokefly.actions import pressed_buttons
from pokefly.red_state import RedState
from pokefly.rom import validate_rom


class RedEmulator:
    def __init__(self, rom: Path, *, headless: bool = True, speed: int = 0) -> None:
        from pyboy import PyBoy

        self.rom_sha1 = validate_rom(rom)
        # File-like ROM avoids implicitly loading or writing adjacent battery saves.
        self._rom = io.BytesIO(rom.read_bytes())
        self.pyboy = PyBoy(self._rom, window="null" if headless else "SDL2", sound_emulated=False)
        self.pyboy.set_emulation_speed(speed)
        self.running = True

    def tick(self, frames: int) -> bool:
        self.running = self.pyboy.tick(frames, True)
        return self.running

    def act(self, action: str, frames: int = 24) -> bool:
        buttons = pressed_buttons(action)
        if frames < 2:
            raise ValueError("An action needs at least two frames for press and release")
        pressed = []
        try:
            for button in buttons:
                self.pyboy.button_press(button)
                pressed.append(button)
            if not self.tick(frames - 1):
                return False
        finally:
            for button in pressed:
                self.pyboy.button_release(button)
        return self.tick(1)  # All buttons released, including between repeated A presses.

    def act_sampled(self, action: str, frames: int, samples: int, observe) -> bool:
        """Same held/released pulse, with raw screen observations during execution.

        Samples are taken AFTER evenly spaced game-frame offsets, including the
        final released frame. The callback can integrate neurons but cannot alter
        the already selected buttons. No game memory is supplied to it.
        """
        buttons = pressed_buttons(action)
        if frames < 2 or not 1 <= samples <= frames:
            raise ValueError("Require >=2 frames and 1..frames visual samples")
        offsets = [(index + 1) * frames // samples for index in range(samples)]
        pressed, elapsed = [], 0
        try:
            for button in buttons:
                self.pyboy.button_press(button)
                pressed.append(button)
            for offset in offsets:
                held_end = min(offset, frames - 1)
                if held_end > elapsed:
                    if not self.tick(held_end - elapsed):
                        return False
                    elapsed = held_end
                if offset == frames:
                    for button in pressed:
                        self.pyboy.button_release(button)
                    pressed.clear()
                    if not self.tick(1):
                        return False
                    elapsed += 1
                observe(self.screen(), offset)
            return True
        finally:
            for button in pressed:
                self.pyboy.button_release(button)

    def bootstrap(self, max_actions: int = 1000) -> int:
        """Explicit scripted intro only; learning begins once gameplay starts."""
        for index in range(max_actions):
            if self.state().started:
                self.tick(90)  # Let the bedroom fade-in finish with buttons released.
                return index
            action = "start" if index % 5 == 0 else "a"
            if not self.act(action, frames=24):
                raise RuntimeError("Emulator closed during the intro")
        raise RuntimeError("Intro did not finish; use 'manual' to create a starting save state")

    def state(self) -> RedState:
        return RedState.read(self.pyboy.memory)

    def screen(self):
        return self.pyboy.screen.ndarray.copy()

    def screenshot(self, path: Path) -> None:
        self.pyboy.screen.image.save(path)

    def save(self, path: Path) -> None:
        with path.open("wb") as stream:
            self.pyboy.save_state(stream)

    def load(self, path: Path, *, advance: bool = True) -> None:
        with path.open("rb") as stream:
            self.pyboy.load_state(stream)
        if advance:
            self.tick(1)

    def close(self) -> None:
        self.pyboy.stop(save=False)
        self._rom.close()

    def __enter__(self) -> RedEmulator:
        return self

    def __exit__(self, *args) -> None:
        self.close()
