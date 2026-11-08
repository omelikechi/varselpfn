# Input preparation tests
import numpy as np
import pandas as pd

from varselpfn.api import covariates_matrix, response_vector

def test_numeric_text_becomes_numeric():
    x = np.random.default_rng(0).standard_normal(50)
    df = pd.DataFrame({'a': [f'{v:.4f}' for v in x], 'b': [None if i % 5 == 0 else f'{v:.4f}' for i, v in enumerate(x)]})
    M, _ = covariates_matrix(df)
    assert np.allclose(M[:, 0], x, atol=1e-4)
    assert np.isfinite(M).all() and np.ptp(M[:, 1]) > 0

def test_identifier_is_constant():
    M, _ = covariates_matrix(pd.DataFrame({'id': [f'id{i}' for i in range(30)]}))
    assert np.ptp(M[:, 0]) == 0

def test_singleton_categories_pooled_apart_from_repeated_ones():
    # 'a' in half the rows, a different one-off value in each of the others: the column must keep 'a' versus the rest
    M, _ = covariates_matrix(pd.DataFrame({'mixed': ['a' if i % 2 else f'v{i}' for i in range(40)]}))
    assert len(np.unique(M[:, 0])) == 2

def test_categorical_dtype_stays_categorical():
    M, _ = covariates_matrix(pd.DataFrame({'c': pd.Categorical(['10', '2', '10', '2', '3', '3'])}))
    assert len(np.unique(M[:, 0])) == 3

def test_missing_and_infinite_filled():
    M, _ = covariates_matrix(np.array([[1.0, np.nan], [2.0, np.inf], [3.0, 5.0], [4.0, 7.0]]))
    assert np.isfinite(M).all() and M[0, 1] == M[1, 1] == 6.0

def test_response_types():
    assert response_vector(np.array([0.1, 2.0, 3.5]))[1] == 'regression'
    assert response_vector(np.array([1, 2, 1, 2]))[1] == 'binary'
    assert response_vector(np.array(['x', 'y', 'z', 'x']))[1] == 'multiclass'
    assert response_vector(np.array([True, False, True]))[1] == 'binary'
    assert response_vector(np.array([0, 1, 2, 1]), 'multiclass')[1] == 'multiclass'
