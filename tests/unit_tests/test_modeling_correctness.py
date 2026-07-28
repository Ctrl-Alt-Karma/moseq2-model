"""
Regression tests for modeling correctness in moseq2_model.

These cover defects that silently changed what the model was fit on, or that
made a fit unreproducible, rather than raising an error.
"""
import numpy as np
from unittest import TestCase
from collections import OrderedDict

from moseq2_model.helpers.data import get_training_data_splits
from moseq2_model.util import get_parameter_strings


def _data_dict(n_sessions=3, n_frames=1000, n_pcs=4, seed=0):
    rng = np.random.RandomState(seed)
    return OrderedDict(
        (f's{i}', rng.randn(n_frames, n_pcs)) for i in range(n_sessions)
    )


class TestTrainingDataSplits(TestCase):
    def test_split_uses_every_frame_exactly_once(self):
        # Both slices previously took the same fraction from opposite ends, so
        # frames in the middle were used by neither for split_frac < 0.5.
        data = _data_dict(n_frames=1000)

        train, val = get_training_data_splits(0.2, data)

        for key in data:
            assert len(train[key]) == 200
            assert len(val[key]) == 800
            assert len(train[key]) + len(val[key]) == len(data[key])

    def test_train_and_validation_do_not_overlap(self):
        # For split_frac > 0.5 the two windows used to overlap, so the reported
        # validation likelihood was computed largely on training frames.
        data = _data_dict(n_frames=1000)

        train, val = get_training_data_splits(0.8, data)

        for key in data:
            assert len(train[key]) == 800
            assert len(val[key]) == 200
            # the validation block must be the tail the model never saw
            np.testing.assert_allclose(val[key], data[key][800:])
            np.testing.assert_allclose(train[key], data[key][:800])

    def test_rejects_degenerate_fractions(self):
        data = _data_dict()
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                get_training_data_splits(bad, data)


class TestKappaScanParameterForwarding(TestCase):
    def _config(self, **overrides):
        config = {
            'npcs': 10,
            'num_iter': 100,
            'index': None,
            'separate_trans': False,
            'robust': False,
            'e_step': False,
            'hold_out': False,
            'nfolds': 5,
            'max_states': 100,
            'ncpus': 0,
            'cluster_type': 'local',
            'nlags': 5,
            'whiten': 'each',
            'alpha': 100.0,
            'gamma': 500.0,
            'var_name': 'my_scores',
            'seed': 7,
        }
        config.update(overrides)
        return config

    def test_forwards_parameters_that_change_the_fit(self):
        # These were dropped, so every model in a kappa scan was fit with the CLI
        # defaults regardless of what the user configured.
        parameters, _ = get_parameter_strings(self._config())

        assert '--nlags 5' in parameters
        assert '--whiten each' in parameters
        assert '--alpha 100.0' in parameters
        assert '--gamma 500.0' in parameters
        assert '--var-name my_scores' in parameters
        assert '--seed 7' in parameters

    def test_omits_absent_parameters(self):
        config = self._config()
        del config['nlags']

        parameters, _ = get_parameter_strings(config)

        assert '--nlags' not in parameters
        # the rest still come through
        assert '--whiten each' in parameters
