from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data, G, WATER_LEVELS
from src.solvers import aggregate_wave_observations, fit_bed


def test_physical_constants_and_full_observation_axes():
    data = load_data()
    assert G == 9.80665 and WATER_LEVELS == {'oblique': .177, 'planview': .183}
    readme = (Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    rows = {}
    for line in readme.splitlines():
        if line.startswith('| '):
            fields = [value.strip() for value in line.strip('|').split('|')]
            if len(fields) == 4:
                rows[fields[0]] = fields
    assert float(rows['Gravitational acceleration'][1]) == G
    assert rows['Gravitational acceleration'][2].startswith('m/s')
    for label in ['oblique', 'planview']:
        title = label.capitalize()
        water = rows[title + ' water-surface elevation']
        assert float(water[1]) == WATER_LEVELS[label] and water[2] == 'm'
        interval = rows[title + ' mean sample interval']
        np.testing.assert_allclose(float(interval[1]), np.mean(np.diff(data['time_' + label])), rtol=0, atol=1e-15)
        assert interval[2] == 's'
    for key, array in data.items():
        row = rows[key]
        assert row[1].replace(' ', '') == str((1, *array.shape)).replace(' ', '')
        assert row[2] == str(array.dtype)
        unit = 'ADU' if key.startswith('signal_') else ('s' if key.startswith('time_') else 'm')
        words = row[3].replace('.', ' ').replace(',', ' ').replace(';', ' ').split()
        assert unit in words
    assert data['signal_oblique'].shape == (5075, 301)
    assert data['signal_planview'].shape == (26382, 300)
    assert data['reconstruction_xy'].shape == (8963, 2)
    for label, count in [('oblique', 301), ('planview', 300)]:
        np.testing.assert_array_equal(data['time_' + label], np.linspace(0, 160, count))


def test_no_observations_means_no_supported_depth():
    obs = {'xy': np.empty((0, 2)), **{k: np.empty(0) for k in ['gamma', 'mean_gamma', 'omega', 'water_level']}}
    prediction, diagnostic = fit_bed(np.array([[0., 0.], [1., 1.]]), obs)
    assert not prediction['validity'].any()
    assert np.equal(prediction['reconstruction'], 0).all()
    assert np.isnan(diagnostic['raw_bed']).all()


def test_aggregation_order_and_observation_only_quality():
    def field(label, omega, mean, std):
        return dict(record=label, key='a', omega=omega, ks=np.array([[omega**2/G]]),
                    mean_gamma=np.array([[mean]]), std_gamma=np.array([[std]]),
                    wave_xy=np.array([[1., 2.]]), water_level=WATER_LEVELS[label])
    result = aggregate_wave_observations([field('oblique', 2., 1., .01),
                                         field('planview', 3., 1., .01)])
    np.testing.assert_array_equal(result['omega'], [3., 2.])
    assert len(aggregate_wave_observations([field('oblique', 2., 1., .08)])['gamma']) == 0
    assert len(aggregate_wave_observations([field('oblique', 2., 1.1, .01)])['gamma']) == 0


def test_bad_raw_shapes_are_rejected(tmp_path):
    raw = {k: v[None] for k, v in load_data().items()}
    raw['signal_oblique'] = raw['signal_oblique'][:, :-1]
    path = tmp_path/'bad.npz'; np.savez_compressed(path, **raw)
    with pytest.raises(ValueError):
        load_data(path)
