from pathlib import Path

N_SIM = 5

def run_experiment(scenario: int, output_dir: Path):
    """Run the experiment for a given scenario.
    
    Args:
        scenario (int): The scenario to run the experiment for.
    
    Returns:
        Dict[str, Any]: The results of the experiment.
    """
    return {
        "scenario": scenario,
        "result": 1
    }