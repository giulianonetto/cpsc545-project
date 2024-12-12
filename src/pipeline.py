from pathlib import Path

def pipeline(output_dir: Path, logger):
    """Pipeline: runs the pipeline for the project.

    Args:
        output_dir (Path): The output directory for the pipeline.
        logger (_type_): The logger object for the pipeline.
    """
    logger.info(f"Starting pipeline with output directory {output_dir.absolute()}")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.info(f"Created output directory {output_dir.absolute()}")
    else:
        logger.warn(f"Output directory {output_dir.absolute()} already exists and the contents will be overwritten.")
    
    
    