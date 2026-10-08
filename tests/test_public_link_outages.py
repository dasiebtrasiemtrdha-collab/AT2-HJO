"""Published blockage knobs and default input compatibility."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from at2hjo.revision4.config import load
from at2hjo.revision4.inputs import Trace


ROOT = Path(__file__).resolve().parents[1]


def test_default_outage_settings_preserve_published_example_link_trace():
    c = load(ROOT / 'configs/revision4/smoke.json')
    trace = Trace(c, 4000)
    with np.load(ROOT / 'data/example/trace.npz', allow_pickle=False) as example:
        for slot in range(c['horizon']):
            # OS math libraries may differ in the last bits of received power.
            # Bandwidth, TX/RX electrical power and availability remain exact.
            expected = example['links'][slot]
            np.testing.assert_array_equal(trace.links[..., 1:], expected[..., 1:])
            np.testing.assert_array_equal(trace.links[..., 0] == 0, expected[..., 0] == 0)
            np.testing.assert_allclose(trace.links[..., 0], expected[..., 0],
                                       rtol=1e-13, atol=0.)
            if slot + 1 < c['horizon']:
                trace.advance()


def test_complete_blockage_disables_bn_rf_links_without_changing_other_inputs():
    c = load(ROOT / 'configs/revision4/smoke.json')
    off = deepcopy(c)
    off['physical']['channels']['bn']['wave_block_probability'] = 1.
    off['physical']['channels']['rf']['nominal_outage_probability'] = 1.
    clear = deepcopy(c)
    clear['physical']['channels']['bn']['wave_block_probability'] = 0.
    clear['physical']['channels']['rf']['nominal_outage_probability'] = 0.
    blocked, available = Trace(off, 4000), Trace(clear, 4000)
    bn = slice(c['nodes'], c['nodes'] + c['buoys'])
    sat = slice(c['nodes'] + c['buoys'], c['nodes'] + c['buoys'] + c['satellites'])
    for slot in range(3):
        assert not blocked.links[bn, bn, 4].any()
        assert not blocked.links[bn, sat, 4].any()
        assert not blocked.links[sat, bn, 4].any()
        assert available.links[bn, bn, 4].sum() == c['buoys'] * (c['buoys'] - 1)
        assert available.links[bn, sat, 4].any()
        np.testing.assert_array_equal(blocked.q, available.q)
        np.testing.assert_array_equal(blocked.fade, available.fade)
        assert blocked.arrivals == available.arrivals
        if slot < 2:
            blocked.advance()
            available.advance()
