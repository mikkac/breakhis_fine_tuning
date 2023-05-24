import os
import time

fold_idx_range = range(5)
learning_rate_values = [1e-6, 1e-5, 3e-5, 1e-4]
weight_decay_values = [0.001, 0.005, 0.01, 0.1]

counter = 300
for learning_rate in learning_rate_values[:1]:
    counter += 1
    for weight_decay_rate in weight_decay_values[:1]:
        for fold_idx in fold_idx_range:
            learning_rate_str = str(learning_rate).replace(".", "_")
            weight_decay_str = str(weight_decay_rate).replace(".", "_")
            experiment_id = f"xxx{counter}_lr_{learning_rate_str}_wd_{weight_decay_str}"
            command = f"python train_models_with_predefined_folds.py {experiment_id} {fold_idx} {learning_rate} {weight_decay_rate}"
            os.system(command)
            time.sleep(5)
