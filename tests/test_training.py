"""Checks for data leakage boundaries and model-selection inputs."""
import unittest
from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train import FEATURES, ROOT, load_data, pipeline, select_model


class TrainingTests(unittest.TestCase):
    def test_targets_and_identifiers_are_not_features(self):
        self.assertFalse(set(FEATURES) & {'codigo', 'nome', 'estado', 'faturamento', 'potencial'})

    def test_imputation_uses_training_median_only(self):
        model = pipeline(DummyRegressor())
        model.fit(pd.DataFrame({'renda': [10., 30., None]}), [1, 2, 3])
        model.predict(pd.DataFrame({'renda': [100000., None]}))
        self.assertEqual(model.named_steps['simpleimputer'].statistics_[0], 20.)

    def test_selection_uses_validation_score(self):
        name, _, rows = select_model(
            {'bad': (DummyRegressor(strategy='constant', constant=100), {}),
             'good': (DummyRegressor(strategy='constant', constant=2), {})},
            pd.DataFrame({'x': range(8)}), [2] * 8,
            'neg_root_mean_squared_error', KFold(2))
        self.assertEqual(name, 'good')
        self.assertEqual(len(rows), 2)

    def test_bundled_data_boundaries(self):
        rj, sp = load_data(ROOT / 'DadosDesafioCientista.xlsx')
        self.assertEqual((len(rj), len(sp)), (160, 296))
        self.assertTrue(sp[['faturamento', 'potencial']].isna().all().all())


if __name__ == '__main__':
    unittest.main()
