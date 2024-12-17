from pathlib import Path
import patchworklib as pw
from .simulation import plot_scenario, SIMULATION_SCENARIOS
from .experiment import run_experiment
import pandas as pd

def step_1(output_dir: Path, logger, overwrite: bool = False):
    """Step 1: generate Figure 1 in the report containing a summary of all simulation scenarios.

    Args:
        output_dir (Path): The output directory for the pipeline.
        logger (_type_): The logger object for the pipeline.
        overwrite (bool, optional): Whether to overwrite the output directory if it exists. Defaults to False.
    """
    logger.info(f"Starting Step 1")
    output_dir = output_dir.joinpath("step_1")
    
    if output_dir.exists():
        if overwrite:
            logger.warn(f"Output directory {output_dir.absolute()} already exists and the contents will be overwritten.")
        else:
            logger.info(f"Output directory {output_dir.absolute()} already exists and the contents will not be overwritten.")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_scenario_plots = []
    for scenario in SIMULATION_SCENARIOS:
        scenario_plots = plot_scenario(scenario=scenario)
        for _, plot in enumerate(scenario_plots):
            all_scenario_plots.append(pw.load_ggplot(plot, figsize=(4,4)))
    
    figure1 = (
        all_scenario_plots[0] | all_scenario_plots[1] | all_scenario_plots[2] | all_scenario_plots[3] | all_scenario_plots[4] 
    ) / (
        all_scenario_plots[5] | all_scenario_plots[6] | all_scenario_plots[7] | all_scenario_plots[8] | all_scenario_plots[9]
    ) / (
        all_scenario_plots[10] | all_scenario_plots[11] | all_scenario_plots[12] | all_scenario_plots[13] | all_scenario_plots[14]
    ) / (
        all_scenario_plots[15] | all_scenario_plots[16] | all_scenario_plots[17] | all_scenario_plots[18] | all_scenario_plots[19]
    )
    
    # figure1 = all_scenario_plots[0] | all_scenario_plots[1] | all_scenario_plots[2] | all_scenario_plots[3] | all_scenario_plots[4]
    
    figure1.savefig(output_dir.joinpath("all_scenarios.png"), dpi=300, quick=True)
    
    
        
def step_2(output_dir: Path, logger, overwrite: bool = False, scenarios: str = "all"):
    """Step 2: generate Figure 2 in the report containing a results of all simulation scenarios.

    Args:
        output_dir (Path): The output directory for the pipeline.
        logger (_type_): The logger object for the pipeline.
        overwrite (bool, optional): Whether to overwrite the output directory if it exists. Defaults to False.
    """
    logger.info(f"Starting Step 2")
    output_dir = output_dir.joinpath("step_2")
    
    if output_dir.exists():
        if overwrite:
            logger.warn(f"Output directory {output_dir.absolute()} already exists and the contents will be overwritten.")
        else:
            logger.info(f"Output directory {output_dir.absolute()} already exists and the contents will not be overwritten.")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if scenarios == "all":
        scenarios = SIMULATION_SCENARIOS
    else:
        scenarios = [int(s) for s in scenarios.split(",")]
    
    all_results = []
    all_plots = []
    for scenario in SIMULATION_SCENARIOS:
        if scenario in scenarios:
            experiment_results = run_experiment(scenario=scenario, output_dir=output_dir)
            all_results.append(experiment_results[0])
            all_plots.append(experiment_results[1])
    
    # save results dataframe to file
    summary_funs = ["mean", "std", "size"]
    summary_funs = {
        "observed_only": summary_funs,
        "predictions_only": summary_funs,
        "pppca": summary_funs
    }
    all_results = pd.concat(all_results).groupby('scenario').agg(summary_funs)
    all_results.reset_index().to_csv(output_dir.joinpath("all_results.csv"), index=False, sep="\t")
    
    _all_results_copy = all_results.copy()
    _all_results_copy.columns = all_results.columns.get_level_values(0)
    _all_results_copy.reset_index().to_csv(output_dir.joinpath("all_results2.csv"), index=False, sep="\t")
    
    # build figure 2 with all four plots
    figure2 = (
        all_plots[0] | all_plots[1]
    ) / (
        all_plots[2] | all_plots[3] 
    )
    
    logger.info(f"Saving Figure 2 to {output_dir.joinpath('all_results.png').absolute()}")
    figure2.savefig(output_dir.joinpath("all_results.png"), dpi=300, quick=True)
    