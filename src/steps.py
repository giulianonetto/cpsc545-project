from pathlib import Path

def step_1(output_dir: Path, logger):
    """Step 1: generate Figure 1 in the report containing a summary of all simulation scenarios.

    Args:
        output_dir (Path): The output directory for the pipeline.
        logger (_type_): The logger object for the pipeline.
    """
    logger.info(f"Starting Step 1")
    output_dir = output_dir.joinpath("step_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    
    