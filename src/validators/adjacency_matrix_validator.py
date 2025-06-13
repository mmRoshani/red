import numpy as np
import pandas as pd

def normalize_values(matrix: np.ndarray) -> np.ndarray:
    numeric_matrix = np.nan_to_num(matrix.astype(float), nan=0.0)
    return np.where(numeric_matrix > 0, 1, 0)

def validate_and_standardize_matrix(file_path: str) -> tuple[np.ndarray, int]:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, header=0)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
    if df.columns[0] == df.iloc[:, 0].name:
        df = df.drop(df.columns[0], axis=1)
    
    matrix = df.to_numpy()
    matrix = normalize_values(matrix)
    
        
    num_nodes = matrix.shape[0]

    return matrix, num_nodes




