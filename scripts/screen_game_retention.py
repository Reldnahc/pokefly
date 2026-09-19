"""Conservative prospective whole-game progress screen; observer only.

Re-audits complete raw panels before scoring. This is an engineering threshold,
not statistical significance, a reward definition, or permission to promote a
model. The same retained lineage must improve on every held-out noise seed.
"""

import argparse
import json
from pathlib import Path

from summarize_game_retention_panel import audit

from pokefly.runner import run_directory, write_json


def arrival_gain(before, after, budget):
    """Missing original arrival is right-censored, never treated as time zero."""
    for value in (before, after):
        if value is not None and (type(value) is not int or not 1 <= value <= budget):
            raise ValueError('Arrival must be a sampled decision inside the evaluation budget')
    reference = budget if before is None else before
    return {
        'original': before, 'retained': after, 'original_right_censored': before is None,
        'gain_fraction_lower_bound': None if after is None else 1 - after / reference,
        'at_least_ten_percent': after is not None and 10 * after <= 9 * reference,
    }


def milestones(row):
    # Rival is a subset of battle wins: compare each, never sum both as victories.
    completed = row['completed_outcomes']
    return {
        'starter': int(row['starter']), 'route1': int(row['route1_visited']),
        **{name: completed[name] for name in ('battle_win', 'rival_win', 'capture', 'gym_win')},
    }


def screen(comparisons, seeds, budget):
    if (type(budget) is not int or budget < 1 or len(seeds) < 2
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Positive budget and at least two distinct held-out seeds required')
    sources = sorted({row['source_launch_seed'] for row in comparisons})
    expected = {(source, seed) for source in sources for seed in seeds}
    if len(sources) < 2:
        raise ValueError('Include all independent training lineages, not a selected winner')
    found, originals, grouped = set(), {}, {source: [] for source in sources}
    for row in comparisons:
        source, seed = row['source_launch_seed'], row['seed']
        key = source, seed
        if key not in expected or key in found:
            raise ValueError('Duplicate or unregistered retained comparison')
        found.add(key)
        original, retained = row['original'], row['retained']
        if originals.setdefault(seed, original) != original:
            raise ValueError('Shared original control differs between retained lineages')
        before, after = milestones(original), milestones(retained)
        grouped[source].append({
            'seed': seed,
            'lost_milestones': [name for name in before if after[name] < before[name]],
            'additional_milestones': [name for name in before if after[name] > before[name]],
            'arrival': {name: arrival_gain(original[name], retained[name], budget)
                        for name in ('first_starter', 'first_battle')},
        })
    if found != expected:
        raise ValueError('A registered retained seed or lineage is missing')
    candidates = []
    for source, rows in grouped.items():
        rows.sort(key=lambda row: row['seed'])
        efficient = [name for name in ('first_starter', 'first_battle')
                     if all(row['arrival'][name]['at_least_ten_percent'] for row in rows)]
        preserved = all(not row['lost_milestones'] for row in rows)
        additional = any(row['additional_milestones'] for row in rows)
        candidates.append({
            'source_launch_seed': source, 'comparisons': rows,
            'all_milestones_preserved': preserved, 'additional_progress': additional,
            'consistently_faster_arrivals': efficient,
            'passed': preserved and additional and bool(efficient),
        })
    return {
        'protocol': 'whole-game-progress-v1', 'steps_per_arm': budget, 'seeds': sorted(seeds),
        'candidates': candidates, 'passed': any(row['passed'] for row in candidates),
        'scope': __doc__, 'independent_confirmation_still_required': True,
        'statistical_significance_claimed': False, 'new_trials': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    validated = audit(args.reports)
    # audit() requires every panel to have the SAME complete per-arm budget.
    budget = json.loads((args.reports[0] / 'report.json').read_text())['steps_per_arm']
    report = {'status': 'completed', 'audit': validated,
              'progress_screen': screen(validated['comparisons'], validated['seeds'], budget)}
    output = run_directory('game-retention-progress-screen')
    write_json(output / 'report.json', report)
    print(json.dumps(report['progress_screen']), flush=True)
    print('Report:', output, flush=True)


if __name__ == '__main__':
    main()
