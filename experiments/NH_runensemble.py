from pathlib import Path
import sys
import os
import yaml

os.chdir(sys.path[0])
sys.path.append('..')
from neuralhydrology.nh_run import start_run, eval_run

# Train model --------
#start_run(config_file=Path("hybrid_extrapolation.yml", gpu=0))
start_run(config_file=Path("hybrid_extrapolation_seed111111.yml", gpu=0))
start_run(config_file=Path("hybrid_extrapolation_seed222222.yml", gpu=0))
start_run(config_file=Path("hybrid_extrapolation_seed333333.yml", gpu=0))
start_run(config_file=Path("hybrid_extrapolation_seed444444.yml", gpu=0))
start_run(config_file=Path("hybrid_extrapolation_seed555555.yml", gpu=0))

# Test model ---------
#run_dir = Path("runs/lstm_extrapolation_0707_175445")
#eval_run(run_dir=Path("runs/hybrid_extrapolation_2609_122616"), period="test")

for file_name in os.listdir("runs"):
    if file_name.startswith('hybrid_extrapolation_seed'):
            config_path = os.path.join("runs", file_name, 'config.yml')

            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
    
            # Update values
            config['predict_last_n'] = 12327
            config['seq_length'] = 12692
        
            # Write the updated values back to the file
            with open(config_path, 'w') as file:
                yaml.safe_dump(config, file)

for file_name in os.listdir("runs"):
    if file_name.startswith('hybrid_extrapolation_seed'):
         eval_run(run_dir=Path("runs/"+file_name), period="test")