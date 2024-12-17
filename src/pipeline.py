from pathlib import Path
from .steps import step_1, step_2

def pipeline(output_dir: Path, logger, overwrite: bool = False,
             steps: str = "all", scenarios: str = "all"):
    """Pipeline: runs the pipeline for the project.

    Args:
        output_dir (Path): The output directory for the pipeline.
        logger (_type_): The logger object for the pipeline.
        overwrite (bool, optional): Whether to overwrite the output directory if it exists. Defaults to False.
    """
    logger.info(f"Starting pipeline with output directory {output_dir.absolute()}")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.info(f"Created output directory {output_dir.absolute()}")
    else:
        logger.warn(f"Output directory {output_dir.absolute()} already exists and the contents will be overwritten.")
    
    
    if "1" in steps or steps == "all":
        step_1(output_dir=output_dir, logger=logger, overwrite=overwrite)
    
    if "2" in steps or steps == "all":
        step_2(output_dir=output_dir, logger=logger, overwrite=overwrite, scenarios=scenarios)