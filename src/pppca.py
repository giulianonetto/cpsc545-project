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

def normalized_frobenius_inner_product(A, B):
    # Compute the Frobenius inner product
    frobenius_inner_product = np.trace(np.dot(A.T, B))
    
    # Compute the Frobenius norms of A and B
    norm_A = np.linalg.norm(A, 'fro')
    norm_B = np.linalg.norm(B, 'fro')
    
    # Compute the normalized Frobenius inner product
    similarity = frobenius_inner_product / (norm_A * norm_B)
    
    return similarity

def get_omega(s_n, s_tilde_n, s_tilde_N, l=None):
    # Compute the matrix omega
    if l is None:
        l = normalized_frobenius_inner_product(s_n, s_tilde_n)
    
    return s_n + l * (s_tilde_N - s_tilde_n)