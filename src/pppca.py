import numpy as np

def get_top_eigenvectors(mat, L=3):
    """Get the top L eigenvectors of a matrix
    corresponding to its L largest eigenvalues.
    If mat is d x d, the output is d x L.
    mat is supposed to be a symmetric matrix (eg, a covariance matrix).
    
    Args:
        mat (np.ndarray): The input matrix.
        L (int, optional): The number of eigenvectors to return. Defaults to 3.

    Returns:
        np.ndarray: The top L eigenvectors of mat.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(mat)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvectors[:, :L]
