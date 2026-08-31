import numpy as np

from gradcell.benchmark.dfn_parameter import (
    BenchmarkFilter,
    apply_multipliers,
    sample_log_multipliers,
    structural_feasibility,
)


def test_single_parameter_sampling_changes_exactly_one_parameter():
    samples = sample_log_multipliers("single", 14, 7, 0.5, 2.0, seed=7)
    assert samples.shape == (14, 7)
    assert np.all(np.sum(~np.isclose(samples, 1.0), axis=1) == 1)
    assert np.array_equal(np.flatnonzero(~np.isclose(samples[0], 1.0)), [0])
    assert np.array_equal(np.flatnonzero(~np.isclose(samples[7], 1.0)), [0])


def test_multi_parameter_sampling_is_reproducible_and_sparse():
    first = sample_log_multipliers("multi", 20, 7, 0.5, 2.0, seed=11)
    second = sample_log_multipliers("multi", 20, 7, 0.5, 2.0, seed=11)
    assert np.array_equal(first, second)
    changed = np.sum(~np.isclose(first, 1.0), axis=1)
    assert np.all((changed >= 2) & (changed <= 4))


def test_parameter_application_clips_volume_fractions():
    nominal = np.asarray([0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0])
    values = apply_multipliers(nominal, np.full((1, 7), 2.0))
    assert np.all(values[0, :5] < 1.0)
    assert np.allclose(values[0, 5:], 2.0)


def test_structural_feasibility_checks_electrode_volume_balance():
    values = np.asarray(
        [
            [0.3, 0.3, 0.4, 0.6, 0.6, 1.0, 1.0],
            [0.5, 0.3, 0.4, 0.6, 0.6, 1.0, 1.0],
        ]
    )
    assert np.array_equal(structural_feasibility(values), [True, False])


def test_filter_rejects_failures_and_near_nominal_cases():
    criterion = BenchmarkFilter(
        min_capacity_change_fraction=0.01, min_voltage_rmse_v=0.005
    )
    nominal_voltage = np.linspace(4.2, 2.5, 20)
    failed = criterion.accepts(0, 5.0, 5.0, nominal_voltage, nominal_voltage)
    near = criterion.accepts(1, 5.0, 5.0, nominal_voltage + 1e-4, nominal_voltage)
    informative = criterion.accepts(1, 4.9, 5.0, nominal_voltage, nominal_voltage)
    assert failed[0] is False and failed[1] == "solver_or_cutoff_failure"
    assert near[0] is False and near[1] == "indistinguishable_from_nominal"
    assert informative[0] is True and informative[1] == "accepted"
