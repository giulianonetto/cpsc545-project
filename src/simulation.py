from typing import Optional
from sklearn.datasets import make_blobs
from .pppca import get_top_eigenvectors
from plotnine import *
import pandas as pd
import numpy as np

SIMULATION_SCENARIOS = [1, 2, 3, 4]
N_SIMULATIONS = 5

def sample_mvnorm(sample_size, number_of_features, rho=0.0, var=2):
    """
    Simulate data from a multivariate normal distribution with a given correlation matrix
    of AR(1) structure. Set rho=0 for independent features. Mean vector always zero.
    
    Args:
        sample_size: int, number of samples
        number_of_features: int, number of features
        rho: float, correlation between features
        var: float, variance of the features
    """
    # Create the AR(1) covariance matrix
    indices = np.arange(number_of_features)
    cor_matrix = rho ** np.abs(np.subtract.outer(indices, indices))
    std_mat = np.diag(np.ones(number_of_features)) * np.sqrt(var)
    cov_matrix = std_mat @ cor_matrix @ std_mat

    # Generate MVN data
    _data = np.random.multivariate_normal(np.zeros(number_of_features), cov_matrix, size=sample_size)
    return _data

def sample_gmm4(sample_size, number_of_features, effect=2, var=2):
    """
    Simulate data from a Gaussian Mixture Model with four components.
    Minimum number of features is 10. Base on sklearn.datasets.make_blobs.
    
    Args:
        sample_size: int, number of samples
        number_of_features: int, number of features
        effect: float, effect size of the cluster means
        var: float, variance of the features
    """
    assert number_of_features >= 10
    n2 = number_of_features // 2
    effect2 = effect / 2
    centers = np.array([
        [+effect if i % 2 == 0 else -effect for i in range(number_of_features)],
        [-effect if i % 2 == 0 else +effect for i in range(number_of_features)],
        [+effect2 if i % 2 == 0 else -effect2 for i in range(n2)] + [0] * (number_of_features  - n2),
        [-effect2 if i % 2 == 0 else +effect2 for i in range(n2)] + [0] * (number_of_features  - n2),
    ])
    _data, _ = make_blobs(n_samples=sample_size, centers=centers, cluster_std=np.sqrt(var))
    return _data

def sample_features(n: int, N: int, d: int, var_g: float = 5.0,
                    var_e: float = 1.0, effect_e: float = 1.0,
                    complete_data: bool = False,
                    center: bool = True,
                    seed: Optional[int] = None):
    """Simulate a dataset with n + N observations and d features.
    
    Returns observed features X for n observations and
    predicted features X_tilde for n+N observations
    (unless complete_data is True, in which case it returns
    all n+N for X and X_tilde).
    
    Args:
        n (int): The number of observations for the observed features.
        N (int): The number of observations for the predicted features.
        d (int): The dimensionality of the features.
        var_g (float): The variance of the G component of X.
        var_e (float): The variance of the E component of X.
        effect_e (float): The effect size of clusters from the E component of X.
        complete_data (bool): Whether to return all n+N observations for X and X_tilde. Defaults to False.
        center (bool): Whether to center the data. Defaults to True.
        seed (int): The random seed for reproducibility.
        
    Returns:
        pd.DataFrame: The simulated dataset.
    """
    if seed is not None:
        np.random.seed(seed)
    
    sample_size = n + N
    
    # Step 1: Simulate G and E
    G = sample_mvnorm(sample_size=sample_size, number_of_features=d, var=var_g)
    E = sample_gmm4(sample_size=sample_size, number_of_features=d, effect=effect_e, var=var_e)
    
    X = G + E
    X_tilde = G
        
    if not complete_data:
        X = X[:n,:]
    
    if center:
        X = X - X.mean(axis=0)
        X_tilde = X_tilde - X_tilde.mean(axis=0)
    
    return X, X_tilde

def simulate_data(n: int, N: int, scenario: int = 1, complete_data: bool = False):
    """Simulate data for the project.
    
    Args:
        n (int): The number of observations for the observed features.
        N (int): The number of observations for the predicted features.
        scenario (int): Scenario number. Defaults to 1.
        complete_data (bool): Whether to return all n+N observations for X and X_tilde. Defaults to False.
    """

    match scenario:
        case 1:
            # strong cluster effect, accurate prediction
            X, X_tilde = sample_features(n=n, N=N, d=1000, var_g=5.0, var_e=1.0, effect_e=1.0, complete_data=complete_data)
        case 2:
            # strong cluster effect, inaccurate prediction
            X, X_tilde = sample_features(n=n, N=N, d=1000, var_g=1.0, var_e=5.0, effect_e=1.0, complete_data=complete_data)
        case 3:
            # weak cluster effect, accurate prediction
            X, X_tilde = sample_features(n=n, N=N, d=1000, var_g=5.0, var_e=1.0, effect_e=0.001, complete_data=complete_data)
        case 4:
            # weak cluster effect, inaccurate prediction
            X, X_tilde = sample_features(n=n, N=N, d=1000, var_g=1.0, var_e=5.0, effect_e=0.001, complete_data=complete_data)
        case _:
            raise ValueError(f"Invalid scenario number={scenario}")
    
    return X, X_tilde

def plot_scenario(scenario: int):
    """Plot a summary of a simulation scenario.
    
    Args:
        scenario (int): The scenario number.
    """
    x, x_tilde = simulate_data(n=1000, N=1000, scenario=scenario, complete_data=True)
    cor_x = pd.DataFrame(np.corrcoef(x, rowvar=False)[:50, :50]).reset_index().melt(id_vars="index")
    cor_x_tilde = pd.DataFrame(np.corrcoef(x_tilde, rowvar=False)[:50, :50]).reset_index().melt(id_vars="index")
    # reverse order of variable
    cor_x["variable"] = pd.Categorical(cor_x["variable"], categories=cor_x["variable"].unique()[::-1])
    cor_x_tilde["variable"] = pd.Categorical(cor_x_tilde["variable"], categories=cor_x_tilde["variable"].unique()[::-1])
    corplot_x = ggplot(
        cor_x,
        aes(x="index", y="variable", fill="value")
    ) + \
        geom_tile() + \
        theme_void(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3),
            legend_key_height=600,
            legend_key_width=60,
        ) + \
        labs(fill="Corr($X$)")
    corplot_x_tilde = ggplot(
        cor_x_tilde,
        aes(x="index", y="variable", fill="value")
    ) + \
        geom_tile() + \
        theme_void(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3),
            legend_key_height=600,
            legend_key_width=60,
        ) + \
        labs(fill="Corr($\\widetilde{X}$)")
    
    cors = []
    for i in range(x.shape[1]):
        cors.append(np.corrcoef(x[:, i], x_tilde[:, i])[0, 1])

    cors = np.array(cors)
    corplot_x_vs_x_tilde = ggplot(pd.DataFrame({"cors": cors}), aes(x="cors")) + \
        geom_histogram(bins=30) + \
        labs(x="Corr($X$, $\\tilde{X}$)", y="Count") + \
        coord_cartesian(xlim=(0, 1)) + \
        theme_classic(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3)
        )
    
    # plot the two PCAs
    w = get_top_eigenvectors(np.cov(x, rowvar=False), L=2)
    w_tilde = get_top_eigenvectors(np.cov(x_tilde, rowvar=False), L=2)
    z = x @ w
    z_tilde = x_tilde @ w_tilde
    z = pd.DataFrame(z, columns=["z1", "z2"])
    z_tilde = pd.DataFrame(z_tilde, columns=["z1", "z2"])
    pca_plot_x = ggplot(z, aes(x="z1", y="z2")) + \
        geom_point() + \
        theme_classic(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3),
        ) + \
        labs(x="PC1 of $X$", y="PC2 of $X$")
    pca_plot_x_tilde = ggplot(z_tilde, aes(x="z1", y="z2")) + \
        geom_point() + \
        theme_classic(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3),
        ) + \
        labs(x="PC1 of of $\\widetilde{X}$", y="PC2 of $\\widetilde{X}$")
    return corplot_x, corplot_x_tilde, corplot_x_vs_x_tilde, pca_plot_x, pca_plot_x_tilde