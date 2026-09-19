"""Audit additional whole-game practice against original AND own earlier brain.

Parent and child are related checkpoints, not independent training lineages.
All registered families and controls are required. Observer only: no rewards,
model edits, new trials, checkpoint selection or significance claim.
"""

import argparse
import json
from pathlib import Path

from screen_game_retention import arrival_gain, milestones, screen
from summarize_game_retention_panel import audit

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.runner import run_directory, write_json


def parse_family(value):
    try:
        parent, child = map(int, value.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use parent_launch_seed:child_launch_seed') from exc
    return parent, child


def screen_extension(comparisons, seeds, budget, families):
    if (len(families) < 2 or any(len(pair) != 2 for pair in families)
            or any(type(seed) is not int for pair in families for seed in pair)):
        raise ValueError('At least two explicit parent/child training families required')
    identifiers = [seed for pair in families for seed in pair]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Parent and child identities must be distinct across families')
    if set(identifiers) != {row['source_launch_seed'] for row in comparisons}:
        raise ValueError('Include every registered parent and child, with no extra sources')
    baseline = screen(comparisons, seeds, budget)
    original_checks = {row['source_launch_seed']: row for row in baseline['candidates']}
    observed = {(row['source_launch_seed'], row['seed']): row['retained']
                for row in comparisons}
    results = []
    for parent, child in families:
        rows = []
        for seed in sorted(seeds):
            before, after = observed[parent, seed], observed[child, seed]
            old, new = milestones(before), milestones(after)
            rows.append({
                'seed': seed,
                'lost_parent_milestones': [key for key in old if new[key] < old[key]],
                'additional_parent_milestones': [key for key in old if new[key] > old[key]],
                'parent_relative_arrival': {
                    key: arrival_gain(before[key], after[key], budget)
                    for key in ('first_starter', 'first_battle')
                },
            })
        preserved = all(not row['lost_parent_milestones'] for row in rows)
        additional = any(row['additional_parent_milestones'] for row in rows)
        efficient = [key for key in ('first_starter', 'first_battle')
                     if all(row['parent_relative_arrival'][key]['at_least_ten_percent']
                            for row in rows)]
        original = original_checks[child]
        results.append({
            'parent_source_launch_seed': parent, 'child_source_launch_seed': child,
            'child_original_comparison': original, 'parent_comparisons': rows,
            'all_parent_milestones_preserved': preserved,
            'additional_parent_progress': additional,
            'consistently_faster_parent_relative_arrivals': efficient,
            'passed': original['passed'] and preserved and (additional or bool(efficient)),
        })
    return {
        'protocol': 'whole-game-training-extension-v1', 'scope': __doc__,
        'steps_per_arm': budget, 'seeds': sorted(seeds), 'families': results,
        'passed': any(row['passed'] for row in results), 'new_trials': False,
        'related_checkpoint_families': len(families),
        'independent_confirmation_still_required': True,
        'statistical_significance_claimed': False,
    }


def verify_parent_checkpoint(parent, child):
    """Follow the raw-audited child's exact links to its assigned pinned parent.

    The caller's panel audit already validates complete sources, model/ROM
    identity and whole-game ancestry. This adds the specific family relation;
    matching model names or last launch seeds is insufficient.
    """
    target = Path(parent['checkpoint']).resolve()
    path = Path(child['run']).resolve()
    visited, links = set(), []
    while True:
        if path in visited:
            raise ValueError('Cyclic training-extension ancestry')
        visited.add(path)
        options = json.loads((path / 'config.json').read_text())['options']
        sources = [options.get(key) for key in ('weights', 'resume') if options.get(key)]
        if not sources:
            raise ValueError('Child does not descend from the assigned parent checkpoint')
        if len(sources) != 1:
            raise ValueError('Ambiguous training-extension parent link')
        _, saved = read_checkpoint(Path(sources[0]))
        checkpoint = Path(saved['directory']).resolve()
        links.append({'run': str(path), 'parent_checkpoint': str(checkpoint)})
        if checkpoint == target:
            if sha256(checkpoint / 'brain.npz') != parent['brain_sha256']:
                raise ValueError('Assigned parent neural checksum changed')
            return {'parent': parent['source_launch_seed'], 'child': child['source_launch_seed'],
                    'exact_parent_checkpoint_verified': True, 'links': links}
        if checkpoint.parent.name != 'checkpoints':
            raise ValueError('Missing original run for training-extension ancestry')
        path = checkpoint.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--families', nargs='+', type=parse_family, required=True,
                        help='Predeclared parent_launch_seed:child_launch_seed pairs')
    args = parser.parse_args()
    validated = audit(args.reports)
    budget = json.loads((args.reports[0] / 'report.json').read_text())['steps_per_arm']
    result = screen_extension(validated['comparisons'], validated['seeds'], budget, args.families)
    sources = {source['source_launch_seed']: source for source in validated['sources']}
    relationships = [verify_parent_checkpoint(sources[parent], sources[child])
                     for parent, child in args.families]
    output = run_directory('game-training-extension-screen')
    write_json(output / 'report.json', {
        'status': 'completed', 'audit': validated, 'extension_screen': result,
        'verified_parent_links': relationships,
    })
    print(json.dumps(result), flush=True)
    print('Report:', output, flush=True)


if __name__ == '__main__':
    main()
