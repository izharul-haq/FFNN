import numpy as np

def unilabel_to_multilabel(unilabel: np.ndarray, unique_class: int) -> np.ndarray:
    """Convert unilabel classification into multilabel
    classification.

    For example, unilabel [0, 1, 2] will be converted
    into [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    Parameters
    ----------
    `unilabel` : ndarray,
        (1D) array of unilabel for every instances. Label
        must be integer
    
    `unique_class : int,
        number of unique class
    
    Returns
    -------
    ndarray,
        (2D) array of multilabel for every instances
    """
    size = len(unilabel)
    res = np.zeros((size, unique_class))

    for i in range(size):
        res[i][unilabel[i]] = 1
    
    return res

def multilabel_to_unilabel(multilabel: np.ndarray) -> np.ndarray:
    """Convert multilabel classification into unilabel
    classification.

    For example, multilabel [0, 1, 2] [[1, 0, 0],
    [0, 1, 0], [0, 0, 1]] will be converted into
    [0, 1, 2]

    Parameters
    ----------
    `multilabel` : ndarray,
        (2D) array of multilabel for every instances.
        Label must be array of integer

    Returns
    -------
    ndarray,
        (1D) array of unilabel for every instances
    """
    return np.where(multilabel == 1)[1]