import numpy as np
import scipy.stats
import numba

def sample_entropy(z):
    """Compute entropy of categorical variable, from samples"""
    vals, counts = np.unique(z, return_counts=True)
    p_x = counts/len(z)
    return scipy.stats.entropy(p_x)

@numba.njit(cache=True,nogil=True)
def sample_mutual_information_numba(x, y):
    """Compute mutual information, aka 'information gain' between two integer valued random samples

    Assume input is integer numpy arrays

    Returns:
        The mutual information as a scalar, measured in Nats
    
    Notes:
        Amazingly, my own crosstab computation seems to be faster than using sklearn.metrics.mutual_info_score
        directly on the samples
    """
    J=x.max()+1
    K=y.max()+1
    N = len(x)
    if J == 1 or K == 1:
        return 0

    p_xy1 = np.bincount(x*K+y)
    if p_xy1.size == J*K:
        p_xy2 = p_xy1
    elif p_xy1.size < J*K:
        p_xy2 = np.zeros(J*K,dtype=p_xy1.dtype)
        p_xy2[:len(p_xy1)] = p_xy1
    
    p_xy=p_xy2.reshape(J,K) / N
    p_x = p_xy.sum(0)
    p_y = p_xy.sum(1)
    mi = 0
    for j in range(J):
        for k in range(K):
            if p_xy[j,k] != 0:
                mi+= np.log(p_xy[j,k]/(p_x[k]*p_y[j]) )*p_xy[j,k]
    return mi
