from correlation_based_feature_selection.information_measures import sample_entropy, sample_mutual_information
import numpy as np


def test_mutual_information():
    assert sample_mutual_information(np.array([0,0,1,1]),np.array([0,1,0,1])) ==  0.0
    assert sample_mutual_information(np.array([0,1,0,1]),np.array([0,1,0,1])) == 0.6931471805599453

def test_entropy():
    assert sample_entropy([1]) == 0
    assert sample_entropy([1,2]) ==  0.6931471805599453

