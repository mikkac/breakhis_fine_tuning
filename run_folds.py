import os
import time

fold_idx_range = range(5)
# learning_rate_values = [3e-7, 5e-7, 1e-6, 1e-5, 3e-5]
# weight_decay_values = [3e-4, 5e-4, 1e-3, 5e-3]
learning_rate_values = [1e-6]
weight_decay_values = [1e-3]

for learning_rate in learning_rate_values:
    for weight_decay_rate in weight_decay_values:
        learning_rate_str = str(learning_rate).replace(".", "_")
        weight_decay_str = str(weight_decay_rate).replace(".", "_")
        experiment_id = f"lr_{learning_rate_str}_wd_{weight_decay_str}"
        for fold_idx in fold_idx_range:
            command = f"python run_fold.py --experiment_id {experiment_id} --fold_idx {fold_idx} --weight_decay_rate {weight_decay_rate} --learning_rate {learning_rate}"
            os.system(command)
            time.sleep(5)
