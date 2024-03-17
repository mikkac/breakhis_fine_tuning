import os
import argparse

# Define the expected structure for each model directory
expected_structure = {
    "files": [
        "all_results.json",
        "config.json",
        "eval_results.json",
        "model.safetensors",
        "trainer_state.json",
        "training_args.bin",
        "train_results.json",
    ],
    "dirs": ["runs"],
}

# Define the expected files in a checkpoint directory
checkpoint_files = [
    "config.json",
    "model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
]


def verify_model_directory(model_path):
    missing_elements = {"files": [], "dirs": [], "checkpoint": []}
    for f in expected_structure["files"]:
        if not os.path.exists(os.path.join(model_path, f)):
            missing_elements["files"].append(f)
    for d in expected_structure["dirs"]:
        dir_path = os.path.join(model_path, d)
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            missing_elements["dirs"].append(d)
    # Check for checkpoint directory
    checkpoint_dir = None
    for item in os.listdir(model_path):
        if item.startswith("checkpoint-") and os.path.isdir(
            os.path.join(model_path, item)
        ):
            checkpoint_dir = os.path.join(model_path, item)
            break
    if checkpoint_dir:
        for f in checkpoint_files:
            if not os.path.exists(os.path.join(checkpoint_dir, f)):
                missing_elements["checkpoint"].append(f)
    else:
        missing_elements["dirs"].append("checkpoint-x")
    return missing_elements


def verify_directories(root_dir, prefixes):
    missing_structure = {}
    for root, dirs, files in os.walk(root_dir):
        dir_name = os.path.basename(root)
        for prefix in prefixes:
            if dir_name.startswith(
                prefix
            ):  # Check if the directory name starts with the prefix
                missing_structure[root] = []
                for i in range(5):  # Expecting model_0 to model_4
                    model_dir = os.path.join(root, f"model_{i}")
                    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
                        # Record missing model directory in a structured manner
                        missing_structure[root].append(
                            {"model": i, "missing": "directory missing"}
                        )
                        continue  # Skip further checks if model directory is missing
                    missing_elements = verify_model_directory(model_dir)
                    if any(missing_elements.values()):  # If any missing element
                        missing_structure[root].append(
                            {"model": i, "missing": missing_elements}
                        )
    return missing_structure


def main():
    parser = argparse.ArgumentParser(description="Verify the structure of directories.")
    parser.add_argument(
        "root_directory", type=str, help="Root directory to start verification from."
    )
    args = parser.parse_args()

    prefixes = ["400x_", "200x_", "100x_", "40x_"]
    print(f"Starting verification in directory {args.root_directory}...")
    missing_structure = verify_directories(args.root_directory, prefixes)

    # Print missing elements
    for dir_path, missing_info in missing_structure.items():
        if missing_info:
            print(dir_path.split('/')[-1])
            for item in missing_info:
                print(item)
            print()


if __name__ == "__main__":
    main()
