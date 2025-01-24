import logging
import os
import shutil
from pathlib import Path


def cleanup_data_directories():
    """Clean up all data directories before starting fresh."""
    logger = logging.getLogger(__name__)

    # Directories to clean
    dirs_to_clean = ["db/mm_index", "db/datafiles", "db/video"]

    try:
        logger.info("Starting cleanup of data directories...")

        for dir_path in dirs_to_clean:
            path = Path(dir_path)
            if path.exists():
                logger.info(f"Cleaning directory: {dir_path}")
                shutil.rmtree(path)
                # Recreate empty directory
                path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"Creating directory: {dir_path}")
                path.mkdir(parents=True, exist_ok=True)

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise
