from .utils import get_logger
from .utils.run_scripts import run_script

get_logger()
logger = get_logger()

script_order = [
    "train_model.py",
    "train_lora.py",
    "evaluate_model.py",
    "evaluate_lora.py"
]

for script in script_order:
    run_script(script)

logger.info("All scripts executed successfully.")