from dataclasses import replace

import pytest

from pokefly.red_state import EVENT_FLAG_BYTES, RedState
from pokefly.rewards import GeneralRewards, RewardConfig


def state(**kwargs):
    initial = RedState(True, 38, 3, 6, 0, 1, 5, 0, bytes(EVENT_FLAG_BYTES))
    return replace(initial, **kwargs)


def battle(*, trainer=0, battle_type=0, link=0, gym=0):
    memory = bytearray(0x10000)
    memory[0xD057] = 2 if trainer else 1
    memory[0xD031] = trainer
    memory[0xD05A] = battle_type
    memory[0xD05C] = gym
    memory[0xD12B] = link
    memory[0xD163] = 1
    memory[0xD16D] = 20  # One surviving player Pokemon.
    return memory


def test_novelty_is_general_not_house_specific_and_persists():
    r = GeneralRewards()
    r.baseline(state())
    r.visit(state())
    assert r.drain() == (0, [])
    r.visit(state(x=4))
    assert r.drain()[0] == r.config.new_tile
    r.visit(state(map_id=0))  # Pallet earns exactly the same categories as any other new map.
    reward, events = r.drain()
    assert reward == r.config.new_area + r.config.new_tile
    assert {e["category"] for e in events} == {"new_area", "new_tile"}
    clone = GeneralRewards()
    clone.restore(r.state())
    clone.visit(state(map_id=0))
    assert clone.drain() == (0, [])
    clone.visit(state(map_id=17))
    assert clone.drain()[0] == reward


def test_no_time_button_party_level_or_raw_event_rewards():
    r = GeneralRewards()
    r.baseline(state())
    r.visit(state(party_count=2, level_sum=100, events=bytes([255]) * EVENT_FLAG_BYTES))
    assert r.drain() == (0, [])
    r.visit(state(map_id=0, battle=1))
    r.visit(state(map_id=0, started=False))
    assert r.drain() == (0, [])


@pytest.mark.parametrize("result", [0, 1, 2])
def test_battle_end_alone_is_never_a_win(result):
    r, m = GeneralRewards(), battle()
    r.event("start", m)
    m[0xCF0B] = result
    r.event("end", m)
    assert r.drain() == (0, [])


@pytest.mark.parametrize(
    "trainer,extra",
    [
        (0, None),
        (1, None),
        (0x22, "gym_win"),
        (0x19, "rival_win"),
        (0x2A, "rival_win"),
        (0x2B, "rival_win"),
    ],
)
def test_completed_victory_categories_pay_once(trainer, extra):
    r, m = GeneralRewards(), battle(trainer=trainer)
    r.event("start", m)
    r.event("trainer_win" if trainer else "faint", m)
    r.event("end", m)
    _, events = r.drain()
    assert [e["category"] for e in events] == ["battle_win"] + ([extra] if extra else [])
    r.event("end", m)
    assert r.drain() == (0, [])


@pytest.mark.parametrize("survivor,result", [(False, 0), (True, 1), (True, 2)])
def test_faint_is_not_a_win_if_player_loses_or_flees(survivor, result):
    r, m = GeneralRewards(), battle()
    r.event("start", m)
    r.event("faint", m)
    m[0xD16D], m[0xCF0B] = int(survivor), result
    r.event("end", m)
    assert r.drain() == (0, [])


@pytest.mark.parametrize("party,battle_type", [(1, 0), (6, 0), (1, 2)])
def test_capture_cleared_before_end_including_pc_and_safari(party, battle_type):
    r, m = GeneralRewards(), battle(battle_type=battle_type)
    r.event("start", m)
    m[0xD163] = party
    m[0xD11C] = 42
    r.event("ball_done", m)
    r.event("ball_done", m)  # Same event cannot pay twice.
    m[0xD11C], m[0xCF0B] = 0, 2
    r.event("end", m)
    reward, events = r.drain()
    assert reward == r.config.capture
    assert [e["category"] for e in events] == ["capture"]
    assert events[0]["species"] == 42


@pytest.mark.parametrize("battle_type,link", [(1, 0), (0, 4)])
def test_demo_and_link_battles_do_not_pay(battle_type, link):
    r, m = GeneralRewards(), battle(battle_type=battle_type, link=link)
    r.event("start", m)
    m[0xD11C] = 42
    r.event("ball_done", m)
    r.event("faint", m)
    r.event("end", m)
    assert r.drain() == (0, [])


def test_failed_throw_and_missing_encounter_do_not_pay():
    r, m = GeneralRewards(), battle()
    r.event("start", m)
    r.event("ball_done", m)
    r.event("end", m)
    m[0xD11C] = 42
    r.event("ball_done", m)
    r.event("trainer_win", m)
    r.event("end", m)
    assert r.drain() == (0, [])


def test_mid_battle_checkpoint_keeps_capture_but_new_encounter_clears_it():
    r, m = GeneralRewards(), battle()
    r.event("start", m)
    m[0xD11C] = 42
    r.event("ball_done", m)
    clone = GeneralRewards()
    clone.restore(r.state())
    m[0xD11C] = 0
    clone.event("end", m)
    assert clone.drain()[0] == clone.config.capture
    clone.event("start", m)
    clone.event("end", m)
    assert clone.drain() == (0, [])


def test_giovanni_only_gym_when_gym_flag_is_set():
    for gym in (0, 8):
        r, m = GeneralRewards(), battle(trainer=0x1D, gym=gym)
        r.event("start", m)
        r.event("trainer_win", m)
        r.event("end", m)
        assert ("gym_win" in [e["category"] for e in r.drain()[1]]) == bool(gym)
    with pytest.raises(ValueError):
        RewardConfig(capture=-1)


def early_rewards():
    return GeneralRewards(RewardConfig(timing="confirmed-outcome-v2"))


@pytest.mark.parametrize(
    "trainer,extra", [(0, None), (1, None), (0x22, "gym_win"), (0x19, "rival_win")]
)
def test_confirmed_outcome_pays_earlier_once_and_survives_reload(trainer, extra):
    r, m = early_rewards(), battle(trainer=trainer)
    m[0xD016] = 20  # Live battle HP, not merely the cached party HP.
    r.event("start", m)
    hook = "trainer_win" if trainer else "faint"
    r.event(hook, m)
    _, events = r.drain()
    assert [e["category"] for e in events] == ["battle_win"] + ([extra] if extra else [])
    clone = early_rewards()
    clone.restore(r.state())
    clone.event(hook, m)
    clone.event("end", m)
    clone.event("end", m)
    assert clone.drain() == (0, [])
    assert clone.active is None


@pytest.mark.parametrize(
    "live_player,live_enemy,result", [(0, 0, 0), (20, 3, 0), (20, 0, 1), (20, 0, 2)]
)
def test_early_wild_reward_rejects_ambiguous_faint(live_player, live_enemy, result):
    r, m = early_rewards(), battle()
    m[0xD016], m[0xCFE7], m[0xCF0B] = live_player, live_enemy, result
    r.event("start", m)
    r.event("faint", m)
    assert r.drain() == (0, [])


def test_double_ko_uses_completed_outcome_not_stale_cached_hp():
    r, m = early_rewards(), battle()
    r.event("start", m)
    r.event("faint", m)  # Active player is dead although cached HP still says 20.
    assert r.drain() == (0, [])
    m[0xD16D] = 0
    r.event("end", m)
    assert r.drain() == (0, [])
    # A different encounter with a surviving reserve can still win at the end.
    m[0xD163], m[0xD16D + 44] = 2, 10
    r.event("start", m)
    r.event("faint", m)
    assert r.drain() == (0, [])
    r.event("end", m)
    assert r.drain()[0] == r.config.battle_win


def test_trainer_faint_is_not_a_completed_encounter():
    r, m = early_rewards(), battle(trainer=1)
    m[0xD016] = 20
    r.event("start", m)
    r.event("faint", m)
    assert r.drain() == (0, [])
    r.event("trainer_win", m)
    assert r.drain()[0] == r.config.battle_win


@pytest.mark.parametrize("battle_type,link", [(1, 0), (0, 4)])
@pytest.mark.parametrize("trainer", [0, 0x19])
def test_early_outcomes_exclude_demo_and_link_battles(battle_type, link, trainer):
    r, m = early_rewards(), battle(battle_type=battle_type, link=link, trainer=trainer)
    m[0xD016], m[0xD11C] = 20, 42
    r.event("start", m)
    for hook in ("ball_done", "faint", "trainer_win", "end"):
        r.event(hook, m)
    assert r.drain() == (0, [])


@pytest.mark.parametrize("party,battle_type", [(1, 0), (6, 0), (6, 2)])
def test_early_capture_matches_normal_pc_and_safari_outcomes(party, battle_type):
    r, m = early_rewards(), battle(battle_type=battle_type)
    m[0xD163], m[0xD11C] = party, 42
    r.event("start", m)
    r.event("ball_done", m)
    reward, events = r.drain()
    assert reward == r.config.capture and events[0]["species"] == 42
    m[0xD11C], m[0xCF0B] = 0, 2
    r.event("ball_done", m)
    r.event("end", m)
    assert r.drain() == (0, [])


def test_legacy_timing_never_pays_before_end_and_rejects_unknown_timing():
    r, m = GeneralRewards(), battle(trainer=0x19)
    r.event("start", m)
    r.event("trainer_win", m)
    assert r.drain() == (0, [])
    assert "paid" not in r.active
    r.event("end", m)
    assert r.drain()[0] == r.config.battle_win + r.config.rival_win
    with pytest.raises(ValueError, match="timing"):
        RewardConfig(timing="guess")


@pytest.mark.parametrize(
    "count,active,other_alive", [(0, 0, False), (7, 0, False), (1, 1, False), (2, 0, True)]
)
def test_final_faint_rejects_invalid_party_and_surviving_other_enemy(count, active, other_alive):
    r, m = GeneralRewards(RewardConfig(timing="last-faint-v3")), battle(trainer=1)
    m[0xD016], m[0xD89C], m[0xCFE8] = 20, count, active
    m[0xD8A6 + 44] = 10 if other_alive else 0
    r.event("start", m)
    r.event("faint", m)
    assert r.drain() == (0, [])


@pytest.mark.parametrize("count,active", [(1, 0), (3, 1), (6, 5)])
def test_final_trainer_faint_uses_live_current_enemy_not_stale_cached_hp(count, active):
    r, m = GeneralRewards(RewardConfig(timing="last-faint-v3")), battle(trainer=0x19)
    m[0xD016], m[0xD89C], m[0xCFE8] = 20, count, active
    m[0xD8A6 + 44 * active] = 10  # Not yet cleared at faint entry.
    r.event("start", m)
    r.event("faint", m)
    assert [e["category"] for e in r.drain()[1]] == ["battle_win", "rival_win"]
    clone = GeneralRewards(r.config)
    clone.restore(r.state())
    clone.event("trainer_win", m)
    clone.event("end", m)
    assert clone.drain() == (0, [])


def test_final_trainer_double_ko_waits_for_confirmed_victory():
    r, m = GeneralRewards(RewardConfig(timing="last-faint-v3")), battle(trainer=1)
    m[0xD89C], m[0xD163], m[0xD16D + 44] = 1, 2, 20
    r.event("start", m)
    r.event("faint", m)
    assert r.drain() == (0, [])
    r.event("trainer_win", m)
    assert r.drain()[0] == r.config.battle_win
