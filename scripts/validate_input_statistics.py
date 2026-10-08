"""Current-generator validation, separate from historical experimental traces."""
import argparse
import json
from pathlib import Path
import numpy as np
from at2hjo.revision4.config import load, input_identity
from at2hjo.revision4.inputs import Trace


def statistics(config, seed):
    trace = Trace(config, seed, 0)
    present_slots = total = zero = multiple = exits = reentries = 0
    previous = trace.present.copy()
    for t in range(config['horizon']):
        if t:
            trace.advance()
            exits += int(np.sum(previous & ~trace.present))
            reentries += int(np.sum(~previous & trace.present))
        counts = np.array([len(tasks) for tasks in trace.arrivals])[trace.present]
        present_slots += len(counts)
        total += int(counts.sum())
        zero += int(np.sum(counts == 0))
        multiple += int(np.sum(counts > 1))
        previous = trace.present.copy()
    return dict(seed=seed, in_area_node_slots=present_slots,
                requests_per_in_area_node_slot=total/present_slots,
                zero_arrival_percent=100*zero/present_slots,
                multiple_arrival_percent=100*multiple/present_slots,
                exits=exits, reentries=reentries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = load(root/'configs/revision4/formal_resource_only.json')
    result = {'scope': 'current input validation; not historical-table reproduction',
              'episode': 0, 'nodes': 30, 'slots': 500, 'seeds': list(range(20)),
              'sd': 'sample SD of 20 per-seed statistics (ddof=1)', 'traces': {}}
    for scenario in ['regular', 'stochastic_open']:
        config = dict(base, trace=scenario)
        rows = [statistics(config, seed) for seed in range(20)]
        keys = [k for k in rows[0] if k not in ['seed', 'in_area_node_slots']]
        result['traces'][scenario] = {
            'input_identity': input_identity(config), 'rows': rows,
            'summary': {k: {'mean': float(np.mean([r[k] for r in rows])),
                            'sd': float(np.std([r[k] for r in rows], ddof=1))}
                        for k in keys}}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({name: item['summary'] for name, item in result['traces'].items()}))


if __name__ == '__main__':
    main()
