import numpy as np

def create_conf_matrix(actual: np.ndarray, predicted: np.ndarray, unique_class: int) -> np.ndarray:
    """ Create confusion matrix based on actual and predicted
    class. Can only be used for unilabel class.

    Parameters
    ----------
    `actual` : ndarray,
        array of actual class. Must be array
        of integer
    
    `predicted`: ndarray,
        array of predicted class. Must be array
        of integer

    `unique_class` : int
        number of classes

    Returns
    -------
    ndarray,
        confusion matrix
    """
    # Initialize confusion matrix
    conf_matrix = np.zeros((unique_class, unique_class))
    
    # Update confusion matrix value
    for i in range(len(actual)):
        pred = predicted[i]
        actl = actual[i]

        conf_matrix[pred][actl] += 1
    
    return conf_matrix

def accuracy(conf_matrix: np.ndarray) -> float:
    """Calculate predicted class accuracy score.
    
    Parameters
    ----------
    `conf_matrix` : ndarray,
        confusion matrix

    Returns
    -------
    float,
        accuracy score
    """
    diag_sum = np.trace(conf_matrix)
    total_sum = conf_matrix.sum()

    return diag_sum / total_sum

def precision(conf_matrix: np.ndarray) -> np.ndarray:
    """Calculate predicted class precision score for
    each class.
    
    Parameters
    ----------
    `conf_matrix` : ndarray,
        confusion matrix

    Returns
    -------
    ndarray,
        precision score for each class
    """
    true_pos = np.diagonal(conf_matrix)
    row_sum = conf_matrix.sum(axis=1)

    return true_pos / row_sum

def recall(conf_matrix: np.ndarray) -> np.ndarray:
    """Calculate predicted class recall score for
    each class.
    
    Parameters
    ----------
    `conf_matrix` : ndarray,
        confusion matrix

    Returns
    -------
    float,
        recall score for each class
    """
    true_pos = np.diagonal(conf_matrix)
    col_sum = conf_matrix.sum(axis=0)

    return true_pos / col_sum

def f1(conf_matrix: np.ndarray) -> float:
    """Calculate predicted class f1 score for
    each class.
    
    Parameters
    ----------
    `conf_matrix` : ndarray,
        confusion matrix

    Returns
    -------
    ndarray,
        f1 score for each class
    """
    precision_score = precision(conf_matrix)
    recall_score = recall(conf_matrix)

    return 2 * (precision_score * recall_score) / (precision_score + recall_score)