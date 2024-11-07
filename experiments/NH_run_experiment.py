from pathlib import Path
import sys
import os

os.chdir(sys.path[0])
sys.path.append("..")
from neuralhydrology.nh_run import start_run, eval_run

# Train model --------
# start_run(config_file=Path("lstm_extrapolation_seed1.yml", gpu=0))

# Test model ---------
run_dir = Path("C:/Users/acuna/Desktop/hybrid_extrapolation_test")
eval_run(run_dir=run_dir, period="test")
