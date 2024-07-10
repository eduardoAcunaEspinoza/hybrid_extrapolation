from pathlib import Path
import sys
import os

os.chdir(sys.path[0])
sys.path.append('..')
from neuralhydrology.nh_run import start_run, eval_run

# Train model --------
start_run(config_file=Path("lstm_extrapolation_seed1.yml", gpu=0))

# Test model ---------
#run_dir = Path("runs/lstm_extrapolation_seed_100_0807_151158")
#eval_run(run_dir=run_dir, period="test")