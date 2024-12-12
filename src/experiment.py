from pathlib import Path
from .simulation import simulate_data, SIMULATION_SCENARIOS, N_SIMULATIONS
from .pppca import get_top_eigenvectors
import numpy as np
import pandas as pd
from plotnine import *
import patchworklib as pw

def run_experiment(scenario: int, output_dir: Path):
    """Run the experiment for a given scenario.
    
    Args:
        scenario (int): The scenario to run the experiment for.
    
    Returns:
        Dict[str, Any]: The results of the experiment.
    """
    
    assert scenario in SIMULATION_SCENARIOS, f"Invalid scenario number={scenario}"
    
    # generate test data
    n_test, N_test = 10_000, 1 # no need for x_tilde in the test data
    x_test, _ = simulate_data(n=n_test, N=N_test, scenario=scenario, complete_data=True)
    
    
    # set up training data sizes
    n, N = 10, 5_000
    # number of components to extract
    L = 300
    reconstruction_errors = {
        "observed_only": [],
        "predictions_only": [],
        "pppca": []
    }
    for i in range(N_SIMULATIONS):
        if i % 10 == 0:
            print(f"Running simulation {i+1}/{N_SIMULATIONS} for scenario {scenario}")
        
        # generate training data
        x, x_tilde = simulate_data(n=n, N=N, scenario=scenario, complete_data=False)
        
        s_n = np.cov(x, rowvar=False)
        s_tilde_n = np.cov(x_tilde[:n,:], rowvar=False)
        s_tilde_N = np.cov(x_tilde[n:,:], rowvar=False)
        s_tilde_all = np.cov(x_tilde, rowvar=False)
        omega = s_n - s_tilde_n + s_tilde_N
        
        w_obs_only = get_top_eigenvectors(s_n, L=L)
        w_pred_only = get_top_eigenvectors(s_tilde_all, L=L)
        w_pppca = get_top_eigenvectors(omega, L=L)
        
        x_hat_obs_only = (x_test @ w_obs_only) @ w_obs_only.T
        x_hat_pred_only = (x_test @ w_pred_only) @ w_pred_only.T
        x_hat_pppca = (x_test @ w_pppca) @ w_pppca.T
        
        reconstruction_errors["observed_only"].append(
            np.linalg.norm(x_hat_obs_only - x_test)**2 / x_test.shape[0]
        )
        reconstruction_errors["predictions_only"].append(
            np.linalg.norm(x_hat_pred_only - x_test)**2 / x_test.shape[0]
        )
        reconstruction_errors["pppca"].append(
            np.linalg.norm(x_hat_pppca - x_test)**2 / x_test.shape[0]
        )

    reconstruction_errors = pd.DataFrame(reconstruction_errors)
    
    reconstruction_errors["scenario"] = scenario
    
    boxplot = plot_experiment_results(reconstruction_errors=reconstruction_errors, output_dir=output_dir, scenario=scenario)
    
    return reconstruction_errors, boxplot
    
def plot_experiment_results(reconstruction_errors: pd.DataFrame, output_dir: Path, scenario: int):
    """Plot the results of the experiment.
    
    Args:
        reconstruction_errors (pd.DataFrame): The results of the experiment.
        output_dir (Path): The output directory for the pipeline.
        scenario (int): The scenario number.
    """
    # melt reconstruction_errors for plotting
    reconstruction_errors = reconstruction_errors.melt(var_name="method", value_name="error")
    
    # build a boxplot of the reconstruction errors
    boxplot = ggplot(reconstruction_errors.query("method!='scenario'"), aes(x="method", y="error")) + \
        geom_boxplot() + \
        labs(x="Method", y="Reconstruction Error") + \
        theme_classic(base_size=18) + \
        theme(
            plot_background=element_rect(fill='white'),
            figure_size=(4, 3)
        )
    # save plot
    boxplot.save(output_dir.joinpath(f"scenario_{scenario}_experiment_results.png"), dpi=300, verbose=False)
    
    return pw.load_ggplot(boxplot, figsize=(4, 3))