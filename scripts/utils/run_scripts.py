from logger import get_logger
import subprocess
logger = get_logger()

def run_script(script_name):
    """Runs a Python script and logs its execution."""
    logger.info(f"Starting {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Successfully completed {script_name}.")
    else:
        logger.error(f"Error running {script_name}: {result.stderr}")
    logger.info(f"Output from {script_name}:\n{result.stdout}")