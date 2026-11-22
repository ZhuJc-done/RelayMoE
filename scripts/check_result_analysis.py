"""Verify claim selection rejects partial seeds, version mixing and unpaired data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile

from analyze_results import analyze


def fixture(seed, baseline=200, version='a'):
    records = {}
    for method, latency in [('nccl', baseline), ('relay_batch', 100)]:
        records[method] = [dict(step=i, useful_bytes=16000,
            full_communication_us=latency if i else 1, dispatch_us=latency / 2,
            combine_us=latency / 4, prepare_us=latency / 4, planner_us=1,
            selection_us=1, metadata_us=1, materialize_us=1,
            dispatch_algobw_GBs=2, combine_algobw_GBs=4) for i in range(100)]
    return dict(config=dict(name=f'size-t10-s{seed}', tokens=10, seed=seed), complete=True,
                summaries={}, records=records, source_sha256={'backend': version},
                measurement='test fixture')


def main():
    with tempfile.TemporaryDirectory() as name, contextlib.redirect_stdout(io.StringIO()):
        root = Path(name)
        def run():
            return analyze(root, root / 'output', [1, 2, 3, 4, 5])
        for seed in range(1, 5):
            (root / f'{seed}.json').write_text(json.dumps(fixture(seed)))
        partial = run()
        assert not partial['verified_best'], 'Partial seed coverage must not count as verified'
        (root / '5.json').write_text(json.dumps(fixture(5)))
        full = run()
        best = next(r for r in full['verified_best'] if r['metric'] == 'full')
        assert abs(best['speedup_geomean'] - 2) < 1e-12, 'Use medians, not fastest step'
        assert abs(best['latency_reduction_pct'] - 50) < 1e-12
        (root / '5.json').write_text(json.dumps(fixture(5, version='b')))
        assert not run()['verified_best'], 'Never pool different source versions'
        unmatched = fixture(5)
        unmatched['records']['nccl'][20]['useful_bytes'] += 1
        (root / '5.json').write_text(json.dumps(unmatched))
        try:
            run()
        except ValueError as error:
            assert 'Unmatched semantic bytes' in str(error)
        else:
            raise AssertionError('Must reject unpaired workloads')
    print('Result analysis: 4 claim-selection checks passed')


if __name__ == '__main__':
    main()
