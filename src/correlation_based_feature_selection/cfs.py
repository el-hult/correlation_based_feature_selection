import array
import bisect
import copy
import logging
import typing

import entropymdlp as mdlp
import joblib
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import sklearn.base
import sklearn.feature_selection

from .prioqueue import PrioQueue
from .information_measures import sample_entropy, sample_mutual_information

JOBLIB_VERBOSE_LEVEL = 0

logger = logging.getLogger(__name__)


@numba.njit(cache=True, nogil=True)
def sum_submatrix_supertriangle3(idx, matrix):
    """sum the supertriangle of a matrix, after picking a submatrix as per idx

    If the idx-vector is not sorted, the matrix must be symmetric

    Args:
        idx - numpy int array
        matrix - numeric numpy array
    """
    k = len(idx)
    out = 0
    for i in range(k):
        for j in range(i + 1, k):
            out += matrix[idx[i], idx[j]]

    return out


@numba.njit(cache=True, nogil=True)
def get_merit_intarr(
    idx,
    cf_corr,
    ff_corr,
) -> float:
    """Private helper to access precomputed correlation matrices

    If the idx-vector is not sorted, the ff_corr must be symmetric

    Args
        idx - int array for which indices to pick
        cf_corr - vector of correlations between features and class
        ff_corr - supertriangular matrix pairwise featurecorrelations
    """
    k = len(idx)
    if k == 0:
        return -1
    elif k == 1:
        return cf_corr[idx].item()
    else:
        k_times_r_cf = cf_corr[idx].sum()
        k_times__kminusone_time_r_ff = 2 * sum_submatrix_supertriangle3(idx, ff_corr)
        return k_times_r_cf / np.sqrt(k + k_times__kminusone_time_r_ff)


def remove_constant_columns(X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Drop constant columns"""
    is_not_constant = X.var(axis=0) != 0
    submatrix_with_variation = X[:, is_not_constant]
    indices_of_columns_with_variation = np.argwhere(is_not_constant).squeeze()
    logger.debug(f"Keeping {is_not_constant.sum()} of {is_not_constant.size}")
    return submatrix_with_variation, indices_of_columns_with_variation


def discretize(x, y):
    """Apply MDLP of Fayyad and Irani 1993 to x"""
    cp = mdlp.cut_points(x, y)
    return np.digitize(x, cp)


def discretize_each_column(X, y):
    """Apply MDLP of Fayyad and Irani 1993 to each column

    Using Joblib to parallelize over columns, for a speedup.
    """
    logger.debug("Start")
    n_columns = X.shape[1]
    out = joblib.Parallel(n_jobs=-1, backend="threading", verbose=JOBLIB_VERBOSE_LEVEL)(
        joblib.delayed(discretize)(X[:, c], y) for c in range(n_columns)
    )
    logger.debug("Done")
    return np.column_stack(out)


def uncertanity_coefficient(z_j, z_k, entropy_j, entropy_k):
    # gain = sklearn.metrics.mutual_info_score(z_j, z_k)
    gain = sample_mutual_information(z_j, z_k)
    return 2 * gain / (entropy_j + entropy_k)


def uc_corr(X, y):
    """Compute the symmetrical Uncertanity Coefficient between two integer valued series

    Preconditions:
        there are no classes with zero observations in either input vector
        both X and y are integer arrays (representing classes)
    """
    logger.debug("start")
    assert np.issubdtype(X.dtype, np.integer)
    assert np.issubdtype(y.dtype, np.integer)
    assert X.ndim == 2
    assert y.ndim == 1
    N = len(X)
    assert len(y) == N
    M = X.shape[1]
    out = np.full((M, M + 1), np.nan)
    Z = np.column_stack((X, y))

    entropies = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(sample_entropy)(Z[:, j]) for j in range(M + 1)
    )
    logger.debug("UC start")
    ucs = joblib.Parallel(n_jobs=-1, backend="threading", verbose=JOBLIB_VERBOSE_LEVEL)(
        joblib.delayed(uncertanity_coefficient)(
            Z[:, j], Z[:, k], entropies[j], entropies[k]
        )
        for j in range(M)
        for k in range(j + 1, M + 1)
    )

    i = 0
    for j in range(M):
        for k in range(j + 1, M + 1):
            out[j, k] = ucs[i]
            i += 1

    ff_corr = out[:, :-1]
    cf_corr = out[:, -1]
    return ff_corr, cf_corr


def pearson_corr(X, y):
    """Compute the pearson correlations for float features and float/binary outcomes"""
    assert X.ndim == 2
    assert y.ndim == 1
    assert len(X) == len(y)
    # In all my datasets, precomputing correlations have been faster than doing it on demand
    ff_corr = corr = np.corrcoef(X, rowvar=False)
    ff_corr[np.tril_indices_from(corr)] = np.nan
    ff_corr = np.abs(ff_corr)
    assert np.all(np.isfinite(np.triu_indices_from(corr, k=1)))

    corr = scipy.stats.pointbiserialr if set(y) == {0, 1} else scipy.stats.pearsonr

    cf_corr = np.array(
        [corr(y, X[:, col]).correlation for col in range(X.shape[1])],
    )
    cf_corr = np.abs(cf_corr)
    assert np.all(np.isfinite(cf_corr))

    return ff_corr, cf_corr


class CorrelationBasedFeatureSelector(
    sklearn.feature_selection.SelectorMixin, sklearn.base.BaseEstimator
):
    def __init__(self, search_heuristic="BestFirstForwards", correlation_measure="UC"):
        """Correlation based Feature Selection algorithm

        Args:
            search_heuristic:
                the heuristic to use. Not so many are implemented. 
            correlation_measure:
                a string indicating which quality measure to use. See also page 70-72 in [1] or
                the screen shots and usage examples in [3]

        Notes:
            Algorithm is due to [1]
            Implementation inspired by [2]
            The 'authorative' implementation is the one in Weka, component CfsSubsetEval [4]. It seems to switch between different correlation measures, as needed, for each variable.

        References:
            [1] Hall, M. A. (2000). Correlation-based feature selection of discrete and numeric class machine learning.
            [2] https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/
            [3] https://machinelearningmastery.com/perform-feature-selection-machine-learning-data-weka/
            [4] https://git.cms.waikato.ac.nz/weka/weka/-/blob/main/trunk/weka/src/main/java/weka/attributeSelection/CfsSubsetEval.java
        """
        heuristics = {
            "BestFirstForwards": (5, "forward"),
            "BestFirstBackwards": (5, "backward"),
            "StepwiseForwards": (0, "forward"),
            "StepwiseBackwards": (0, "backward"),
        }
        if search_heuristic not in heuristics:
            raise ValueError(
                f"'{search_heuristic} is not a valid heuristic. Choices are {list(heuristics.keys())}"
            )
        self.max_backtrack, self.search_heuristic = heuristics[search_heuristic]

        assert correlation_measure in ["UC", "Pearson"], (
            "Only some attribute quality measures is implemented. See page 71 in the thesis"
        )
        self.correlation_measure = correlation_measure

        self.input_columns = None
        self.n_features_in_: typing.Optional[int] = None
        self.support_: typing.Optional[npt.NDArray[np.bool_]] = None
        self.feature_names_in_: typing.Optional[npt.NDArray[np.string_]] = None

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("You must fit the selector first!")
        return self.support_

    def fit(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
        y: typing.Union[np.ndarray, pd.Series],
    ):
        """
        Args:
            X a data matrix. numpy array or pandas dataframe. Must be 2d.
            y a vector of outcomes. Must be 1d. Same length as X

        Preconditions:
            assumes that labels are binary, and features are continuously distributed (so that pearson rho makes sense...)
        """
        # Make an exhaustive search for best merit regarding a single variable, as a starting point
        logger.debug("Beginning to compute feature selection")
        if isinstance(X, pd.DataFrame):
            self.input_columns = X.columns
            self.feature_names_in_ = X.columns.to_numpy()
            X = X.to_numpy()

        n_features = X.shape[1]
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        assert y.ndim == 1
        assert X.ndim == 2
        assert len(y) == len(X)
        y = y.astype(int)
        if np.all(y == y[0]):
            logger.error(
                "The outcome is constant. It makes no sense. I will crash now."
            )
            raise ValueError("Constant output makes no sense...")

        #
        # Precompute correlation measures
        #
        if self.correlation_measure == "UC":
            X_discretized = discretize_each_column(X, y)
            X_subset, subset_column_numbers = remove_constant_columns(X_discretized)
            ff_corr, cf_corr = uc_corr(X_subset, y)
        elif self.correlation_measure == "Pearson":
            X_subset, subset_column_numbers = remove_constant_columns(X)
            ff_corr, cf_corr = pearson_corr(X_subset, y)
        else:
            raise ValueError(f"Invalid correlation measure {self.correlation_measure}")

        assert np.all(np.isfinite(np.triu_indices_from(ff_corr, k=1))), (
            "Programming error? I cleared out all constant feature-columns, so there should be only finite correlations"
        )
        assert np.all(np.isfinite(cf_corr)), (
            "Programming error? I cleared out all constant feature-columns and checked for constant labels, so there should be only finite correlations"
        )
        logger.debug("Completed correlation computations")

        #
        # Perform the search
        #
        best_columns = best_first(
            self.search_heuristic, cf_corr, ff_corr, self.max_backtrack
        )

        #
        # Save the result and do some book-keeping
        #
        self.column_indices_relative_full_X = subset_column_numbers[best_columns]

        self.support_ = np.zeros(shape=n_features, dtype=bool)
        self.support_[self.column_indices_relative_full_X] = 1
        self.n_features_to_select_ = self.support_.sum()
        self.n_features_in_ = n_features

        logger.info(
            f"Done. Decided to keep {self.n_features_to_select_} features out of {n_features}"
        )
        return self


def best_first(
    mode: typing.Literal["forward", "backward"], cf_corr, ff_corr, max_backtrack
):
    """Best First Search in Forwards Selection mode with `backtracking`

    This implementation is essentially what Mark Hall specifies in his thesis [1] on
    page 31, table 3.2, with the addition from page 70 about backtracking

    References
        [1] https://www.cs.waikato.ac.nz/~mhall/thesis.pdf
    """
    if mode == "forward":
        return bfs_forward(cf_corr, ff_corr, max_backtrack)
    if mode == "backward":
        return bfs_backward(cf_corr, ff_corr, max_backtrack)
    else:
        raise ValueError(f"{mode=} is unknown")


def bfs_forward(cf_corr, ff_corr, max_backtrack):
    """Best First Search in Forwards Selection mode with `backtracking`

    This implementation is essentially what Mark Hall specifies in his thesis [1] on
    page 31, table 3.2, with the addition from page 70 about backtracking

    References
        [1] https://www.cs.waikato.ac.nz/~mhall/thesis.pdf
    """

    n_columns = len(cf_corr)
    n_backtrack = 0

    state = array.array("L")
    neg_merit = -get_merit_intarr(np.array(state), cf_corr, ff_corr)

    open_set: PrioQueue[array.array] = PrioQueue()
    open_or_closed: typing.Set[typing.ByteString[int]] = set()
    open_set.add_task(state, neg_merit)
    open_or_closed.add(bytes(state))

    record_neg_merit = copy.copy(neg_merit)
    record_state = copy.copy(state)

    while True:
        if len(open_set) == 0:
            # no more states to expand
            break

        # pick a state to expand
        state, neg_merit = open_set.pop_task_and_prio()
        if neg_merit < record_neg_merit:
            record_state = copy.copy(state)
            record_neg_merit = copy.copy(neg_merit)

        # expand into all children and record all that are any good
        locally_optimal = True
        for c in range(n_columns):
            c_pos = bisect.bisect_right(state, c)
            if c_pos > 0 and state[c_pos - 1] == c:
                # element is already in list, dont expand this state
                continue

            # lets insert, evaluate this state, and push it on the queue
            state.insert(c_pos, c)
            candidate_neg_merit = -get_merit_intarr(np.array(state), cf_corr, ff_corr)
            as_bytes = bytes(state)
            if as_bytes not in open_or_closed:
                open_or_closed.add(as_bytes)
                open_set.add_task(copy.copy(state), candidate_neg_merit)
            if candidate_neg_merit < neg_merit:
                locally_optimal = False
            state.pop(c_pos)

        # if it was a dead end - do backtrack
        if locally_optimal:
            n_backtrack += 1
            if n_backtrack > max_backtrack:
                break

    return np.array(record_state)


def bfs_backward(cf_corr, ff_corr, max_backtrack):
    """
    Notes:

    """
    n_columns = len(cf_corr)
    n_backtrack = 0

    state = array.array("L", range(n_columns))
    neg_merit = -get_merit_intarr(np.array(state), cf_corr, ff_corr)

    open_set: PrioQueue[array.array] = PrioQueue()
    open_or_closed: typing.Set[typing.ByteString[int]] = set()
    open_set.add_task(state, neg_merit)
    open_or_closed.add(bytes(state))

    record_neg_merit = copy.copy(neg_merit)
    record_state = copy.copy(state)

    while True:
        if len(open_set) == 0:
            # no more states to expand
            break

        # pick a state to expand
        state, neg_merit = open_set.pop_task_and_prio()
        if neg_merit < record_neg_merit:
            record_state = copy.copy(state)
            record_neg_merit = copy.copy(neg_merit)
            logger.debug(
                f"New record of merit {-neg_merit} using {len(record_state)} variables",
            )

        # expand into all children and record all that are any good
        locally_optimal = True
        for c_pos, c in enumerate(state):
            # lets insert, evaluate this state, and push it on the queue
            state.pop(c_pos)
            candidate_neg_merit = -get_merit_intarr(np.array(state), cf_corr, ff_corr)
            as_bytes = bytes(state)
            if as_bytes not in open_or_closed:
                open_or_closed.add(as_bytes)
                open_set.add_task(copy.copy(state), candidate_neg_merit)
            if candidate_neg_merit < neg_merit:
                locally_optimal = False
            state.insert(c_pos, c)

        # if it was a dead end - do backtrack
        if locally_optimal:
            n_backtrack += 1
            if n_backtrack > max_backtrack:
                break

    return np.array(record_state)
