import numpy as np
import pandas as pd
import os
import warnings
#from src.utils.yaml_loader import config

def normalize_values(matrix: np.ndarray) -> np.ndarray:
    numeric_matrix = np.nan_to_num(matrix.astype(float), nan=0.0)
    return np.where(numeric_matrix > 0, 1, 0)

def validate_and_standardize_matrix(file_path: str) -> tuple[np.ndarray, int]:
    if file_path.endswith('.csv'):
        _df = pd.read_csv(file_path, header=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        _df = pd.read_excel(file_path, header=0)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
    if _df.columns[0] == _df.iloc[:, 0].name:
        _df = _df.drop(_df.columns[0], axis=1)
    
    _matrix = _df.to_numpy()
    _matrix = normalize_values(_matrix)
         
    _num_nodes = _matrix.shape[0]
    # if _num_nodes != self.config.number_of_clients:
    #     warnings.warn(f"Number of nodes in adjacency matrix ({_num_nodes}) does not match number of clients ({self.config.number_of_clients})")
    #     warnings.warn(f"Changed number of clients to {_num_nodes}")
    #     self.config.number_of_clients = _num_nodes
    return _matrix, _num_nodes

def validate_adjacency_matrix(adjacency_matrix_file_name: str) -> np.ndarray:
    _adjacency_matrix_project_relative_file_path = validate_adjacency_matrix_file_path_exists(
        create_project_relative_path(adjacency_matrix_file_name)
    )
    _adjacency_matrix = validate_and_standardize_matrix(
        _adjacency_matrix_project_relative_file_path
    )
    if not validate_matrix_is_square(_adjacency_matrix):
        raise ValueError("adjacency_matrix is not square")
    
    if not validate_matrix_is_symmetric(_adjacency_matrix):
        raise ValueError("adjacency_matrix is not symmetric")
    
    return _adjacency_matrix

def create_project_relative_path(adjacency_matrix_file_name: str) -> str:
    if adjacency_matrix_file_name is not None:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'templates', 'topologies_custom', adjacency_matrix_file_name)
    else:
        raise ValueError("adjacency_matrix_file_name is None")
    
def validate_adjacency_matrix_file_path_exists(adjacency_matrix_project_relative_file_path: str) -> str:
    if os.path.exists(adjacency_matrix_project_relative_file_path):
        return adjacency_matrix_project_relative_file_path
    else:
        raise ValueError(f"adjacency_matrix_project_relative_file_path"
                         f"{adjacency_matrix_project_relative_file_path} does not exist")

def validate_matrix_is_square(matrix: np.ndarray) -> bool:
    return matrix.shape[0] == matrix.shape[1]

def validate_matrix_is_symmetric(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, matrix.T)