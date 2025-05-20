import logging
import unittest

import numpy as np
import scipy.special
import sklearn.model_selection
import sklearn.svm

from correlation_based_feature_selection.cfs import (
    CorrelationBasedFeatureSelector,
    best_first,
)

logger = logging.getLogger(__name__)


class TestStuff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Prepare data
        #
        # (manual step) download the MADELON data as an arff file from
        # https://www.openml.org/search?type=data&sort=runs&id=1485
        from sklearn.datasets import fetch_openml

        df = fetch_openml(
            name="madelon", version=1, as_frame=True, parser="pandas"
        ).frame

        label = "Class"
        features = [name for name in df.columns if name != label]
        df[label] = df[label].astype("int") - 1
        df[features] = df[features].astype("float")
        cls.X_madelon = df[features]
        cls.y_madelon = df[label]

    def test_numpy(self):
        """Check that the API works for handcrafted data and numpy arrays."""
        np.random.seed(0)
        n_vars_all = 100
        N = 1000
        X = np.random.standard_normal(size=(N, n_vars_all))
        probs = scipy.special.expit(1.2 * X[:, 1] + 2 * X[:, 0] + X[:, 2])
        y = probs > np.random.random(size=N)

        trans = CorrelationBasedFeatureSelector(correlation_measure="Pearson")
        import sklearn.linear_model

        clf = sklearn.linear_model.LogisticRegression()

        scores = sklearn.model_selection.cross_val_score(
            clf, X, y, cv=5, scoring="accuracy"
        )
        score_all = np.mean(scores)

        X = trans.fit_transform(X, y)
        scores = sklearn.model_selection.cross_val_score(
            clf, X, y, cv=5, scoring="accuracy"
        )
        score_cfs = np.mean(scores)

        np.testing.assert_array_equal(trans.get_feature_names_out(), ["x0", "x1", "x2"])
        self.assertLess(
            len(trans.get_feature_names_out()), n_vars_all, "Should select a subset"
        )
        self.assertGreater(score_cfs, score_all, "The score should increase")

    def madelon(self, correlation_measure, search_heuristic):
        """Run variable selection on the Madelon dataset, and report how many variables were selected etc"""

        if search_heuristic is None:
            X = self.X_madelon
            y = self.y_madelon

            svc = sklearn.svm.SVC(kernel="rbf", C=100, gamma=0.01, random_state=42)
            scores = sklearn.model_selection.cross_val_score(svc, X, y, cv=10)

            return X.columns.to_numpy(), np.mean(scores)
        else:
            trans = CorrelationBasedFeatureSelector(
                search_heuristic=search_heuristic,
                correlation_measure=correlation_measure,
            )

            X = trans.fit_transform(self.X_madelon, self.y_madelon)
            y = self.y_madelon

            svc = sklearn.svm.SVC(kernel="rbf", C=100, gamma=0.01, random_state=42)
            scores = sklearn.model_selection.cross_val_score(svc, X, y, cv=10)

            return trans.get_feature_names_out(), np.mean(scores)

    def test_madelon_0(self):
        """Make sure I've got the reference number correct, i.e. no variable selection."""
        variables, score = self.madelon(None, None)
        self.assertEqual(score, 0.5)
        np.testing.assert_array_equal(variables, self.X_madelon.columns.to_numpy())

    def test_madelon_1(self):
        """See that accuracy improves"""
        variables, score = self.madelon("Pearson", "StepwiseForwards")
        self.assertEqual(score, 0.665)
        self.assertEqual(len(variables), 48)


class TestSearchLogic(unittest.TestCase):
    @staticmethod
    def make_test_data(n):
        np.random.seed(0)
        cf = np.random.random(size=n)
        ff = np.random.random(size=(n, n))
        ff[np.tril_indices_from(ff)] = np.nan
        return cf, ff

    def test_1(self):
        idx = [0]
        cf, ff = self.make_test_data(1)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_2(self):
        idx = [0, 1]
        cf, ff = self.make_test_data(2)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_3(self):
        idx = [0, 1]
        cf, ff = self.make_test_data(3)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_3b(self):
        idx = [0, 1]
        cf, ff = self.make_test_data(3)
        np.testing.assert_array_equal(idx, best_first("backward", cf, ff, 5))

    def test_4(self):
        idx = [2, 3]
        cf, ff = self.make_test_data(4)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_4b(self):
        idx = [2, 3]
        cf, ff = self.make_test_data(4)
        np.testing.assert_array_equal(idx, best_first("backward", cf, ff, 5))

    def test_5(self):
        idx = [1, 4]
        cf, ff = self.make_test_data(5)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_5b(self):
        idx = [1, 4]
        cf, ff = self.make_test_data(5)
        np.testing.assert_array_equal(idx, best_first("backward", cf, ff, 5))

    def test_6(self):
        idx = [1, 2]
        cf, ff = self.make_test_data(6)
        np.testing.assert_array_equal(idx, best_first("forward", cf, ff, 5))

    def test_10(self):
        idx = [3, 7, 8]
        cf, ff = self.make_test_data(10)
        np.testing.assert_array_equal(best_first("forward", cf, ff, 5), idx)

    def test_100(self):
        idx = [7, 20, 23, 38, 52, 66, 74]
        cf, ff = self.make_test_data(100)
        np.testing.assert_array_equal(best_first("forward", cf, ff, 5), idx)

    def test_500(self):
        """This data is manually checked. The test case is mostly for profiling and regression testing"""
        idx = [149, 215, 349, 376, 448]
        cf, ff = self.make_test_data(500)
        np.testing.assert_array_equal(best_first("forward", cf, ff, 5), idx)
