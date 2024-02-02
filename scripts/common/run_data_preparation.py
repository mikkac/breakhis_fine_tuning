import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from PIL import Image, ImageOps, ImageEnhance
import random
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle
import json



preprocessing_scenarios = {
    "patches_fixed": None,
    "patches_fixed_with_random": None,
    "patches_fixed_with_random_with_transformations": None,
    "patches_fixed_with_random_with_filtered_cells": None,
    "patches_fixed_with_random_with_filtered_background": None,
}

chosen_scenario = "patches"
zoom = 400

cwd = Path().absolute()
main_dir = Path().absolute().parent.parent
input_path = main_dir / 'breakhis'
output_path = main_dir / 'data' / f'{zoom}x'
output_path_original = output_path / 'original'

print(f"Current working directory: {cwd}")
print(f"Main directory: {main_dir}")
print(f"Input path: {input_path}")
print(f"Output path: {output_path}")
print(f"Output path original: {output_path_original}")



shutil.rmtree(output_path, ignore_errors=True)


os.makedirs(output_path_original)
os.makedirs(output_path_original / 'benign')
os.makedirs(output_path_original / 'malignant')


input_data_df = pd.read_csv(input_path / 'Folds.csv').rename(columns={"filename": "path"})

input_data_df['filename'] = input_data_df['path'].apply(
    lambda x: x.split("/")[-1])
input_data_df["label"] = input_data_df['path'].apply(lambda x: x.split("/")[3])
input_data_df["patient_id"] = input_data_df['path'].apply(
    lambda x: x.split("/")[-3])

input_data_df = input_data_df[input_data_df.mag == zoom]

for i in range(len(input_data_df)):
    src = input_path / 'BreaKHis_v1' / input_data_df['path'].iloc[i]
    dest = output_path_original / \
        input_data_df["label"].iloc[i] / str(src).split("/")[-1]
    shutil.copyfile(src, dest)

print(f"Benign: {len(os.listdir(output_path_original / 'benign'))}")
print(f"Malignant: {len(os.listdir(output_path_original / 'malignant'))}")


input_data_df['file_loc'] = input_data_df['label'] + \
    "_" + input_data_df['filename']
input_data_df['class'] = input_data_df['label'].apply(
    lambda x: 0 if x == 'benign' else 1)


benign_df = input_data_df[input_data_df['label'] == 'benign']
malignant_df = input_data_df[input_data_df['label'] == 'malignant']

benign_files = os.listdir(output_path_original / 'benign')
benign_files = [f"data/{zoom}x/original/benign/" +
                file_name for file_name in benign_files]
malignant_files = os.listdir(output_path_original / 'malignant')
malignant_files = [f"data/{zoom}x/original/malignant/" +
                   file_name for file_name in malignant_files]

original_df = pd.DataFrame(benign_files + malignant_files).rename(columns={0: 'file_loc'})

original_df['label'] = original_df['file_loc'].apply(
    lambda x: 0 if x.split('/')[-1].split("_")[1] == 'B' else 1)
original_df['label_str'] = original_df['file_loc'].apply(
    lambda x: "benign" if x.split('/')[-1].split("_")[1] == 'B' else "malignant")

original_df['patient_id'] = original_df['file_loc'].apply(
    lambda x: "-".join(x.split("-")[:3]).split("/")[-1])
original_df.set_index("file_loc", inplace=True)

def rotate_image(image):
    return image.rotate(random.choice([0, 90, 180, 270]), expand=True)

def flip_image(image):
    if random.choice([True, False]):
        image = ImageOps.flip(image)
    if random.choice([True, False]):
        image = ImageOps.mirror(image)
    return image

def color_jitter_image(image):
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness]
    for enhancer in enhancers:
        enhancer = enhancer(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))
    return image

def add_noise_image(image):
    array = np.array(image)
    noise = np.random.randint(0, 50, array.shape, dtype='uint8')
    image = Image.fromarray(np.clip(array + noise, 0, 255).astype('uint8'), 'RGB')
    return image

transformations = [rotate_image, flip_image, color_jitter_image, add_noise_image]


def patches(original_df, scenario_name, num_random_patches=5, min_distance=15, transformations=None, predicate=None):
    """
    Update the dataframe to include random patches with transformations and apply a predicate to each patch.

    :param original_df: Original dataframe.
    :param scenario_name: Scenario name.
    :param num_random_patches: Total number of additional random patches to generate from each image.
    :param min_distance: Minimum distance between the starting points of two random patches.
    :param transformations: Optional list of transformations to apply to the patches.
    :param predicate: A function that takes an image patch as input and returns True if the patch should be included.
    :return: Updated dataframe.
    """
    patch_size = 224
    patch_dfs = []

    for idx, row in tqdm(original_df.iterrows(), total=len(original_df), desc=f"Processing Images: {scenario_name}"):
        img = Image.open(str(main_dir / idx))
        width, height = img.size

        # Generate fixed patches
        fixed_patches_count = 0
        random_patches_count = 0
        for x in range(0, width - patch_size + 1, patch_size):
            for y in range(0, height - patch_size + 1, patch_size):
                right, lower = x + patch_size, y + patch_size
                patch = img.crop((x, y, right, lower))

                if transformations:
                    transformation = random.choice(transformations)
                    patch = transformation(patch)

                if predicate is None or predicate(patch):
                    base_filename = os.path.splitext(idx)[0]
                    new_filename = f"{base_filename}_fixed_{fixed_patches_count}.png".replace("original", scenario_name)
                    patch.save(str(main_dir / new_filename))

                    new_df = pd.DataFrame([row.values], columns=row.index, index=[new_filename])
                    patch_dfs.append(new_df)
                    fixed_patches_count += 1

        # Generate random patches
        patch_start_points = [(x, y) for x in range(0, width - patch_size + 1, patch_size) 
                              for y in range(0, height - patch_size + 1, patch_size)]
        total_tries = 100
        while random_patches_count < num_random_patches and total_tries > 0:
            total_tries -= 1
            valid_patch = False
            tries = 300
            while not valid_patch and tries > 0:
                tries -= 1
                stride_x = random.randint(0, width - patch_size)
                stride_y = random.randint(0, height - patch_size)

                if all(abs(stride_x - x) >= min_distance and abs(stride_y - y) >= min_distance for x, y in patch_start_points) and stride_x + patch_size <= width and stride_y + patch_size <= height:
                    valid_patch = True
                    patch_start_points.append((stride_x, stride_y))

                    left = stride_x
                    upper = stride_y
                    right = left + patch_size
                    lower = upper + patch_size
                    patch = img.crop((left, upper, right, lower))

                    if transformations:
                        transformation = random.choice(transformations)
                        patch = transformation(patch)

                    if predicate is None or predicate(patch):
                        new_filename = f"{base_filename}_random_{random_patches_count}.png".replace("original", scenario_name)
                        patch.save(str(main_dir / new_filename))

                        new_df = pd.DataFrame([row.values], columns=row.index, index=[new_filename])
                        patch_dfs.append(new_df)
                        random_patches_count += 1

    df = pd.concat(patch_dfs)
    df.index.name = 'file_loc'
    return df

def filter_patch(image, black_pixels_predicate):
    gray_image = image.convert("L")

    image_np = np.array(gray_image)
    _, thresholded_image = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = thresholded_image.size
    black_pixels = np.sum(thresholded_image == 0)
    return black_pixels_predicate(black_pixels / total_pixels)

preprocessing_scenarios = {
    "patches_fixed": None,
    "patches_fixed_with_random": None,
    "patches_fixed_with_random_with_transformations": None,
    "patches_fixed_with_random_with_filtered_cells": None,
    "patches_fixed_with_random_with_filtered_background": None,
}

preprocessing_scenarios["patches_fixed"] = lambda df, scenario_name: patches(
    original_df=df,
    scenario_name=scenario_name,
    num_random_patches=0,
    transformations=None,
)

preprocessing_scenarios[
    "patches_fixed_with_random"
] = lambda df, scenario_name: patches(
    original_df=df,
    scenario_name=scenario_name,
    num_random_patches=5,
    transformations=None,
)

preprocessing_scenarios[
    "patches_fixed_with_random_with_transformations"
] = lambda df, scenario_name: patches(
    original_df=df,
    scenario_name=scenario_name,
    num_random_patches=5,
    transformations=transformations,
)

preprocessing_scenarios[
    "patches_fixed_with_random_with_filtered_cells"
] = lambda df, scenario_name: patches(
    original_df=df,
    scenario_name=scenario_name,
    num_random_patches=5,
    transformations=None,
    predicate=lambda patch: filter_patch(
        patch, lambda percentage_black_pixels: percentage_black_pixels > 0.5
    ),
)

preprocessing_scenarios[
    "patches_fixed_with_random_with_filtered_background"
] = lambda df, scenario_name: patches(
    original_df=df,
    scenario_name=scenario_name,
    num_random_patches=5,
    transformations=None,
    predicate=lambda patch: filter_patch(
        patch, lambda percentage_black_pixels: percentage_black_pixels < 0.5
    ),
)

datasets = {"original": {"df": original_df, "path": output_path_original}}
for scenario_name, scenario_func in preprocessing_scenarios.items():
    if scenario_func is not None:
        shutil.rmtree(output_path / scenario_name, ignore_errors=True)
        os.makedirs(output_path / scenario_name)
        os.makedirs(output_path / scenario_name / "benign")
        os.makedirs(output_path / scenario_name / "malignant")
        datasets[scenario_name] = {
            "df": scenario_func(original_df, scenario_name),
            "path": output_path / scenario_name,
        }


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def map_nested_indices(nested_indices, original_indices):
    return original_indices[nested_indices]


class StratifiedGroupKFold:
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.used_group_ids = []

    def _fill_bucket(self, bucket, class_counts, group_ids, y):
        for group_id, label in zip(group_ids, y):
            if group_id in self.used_group_ids:
                continue
            if class_counts[label] > 0:
                group_indices = np.where(group_ids == group_id)[0]
                bucket[label].extend(group_indices)
                class_counts[label] -= len(group_indices)
                self.used_group_ids.append(group_id)

    def _create_buckets(self, group_ids, y, class_ratios):
        total_samples = len(group_ids)
        samples_per_split = total_samples // self.n_splits

        buckets = []
        for _ in range(self.n_splits):
            bucket = {label: [] for label in class_ratios.keys()}
            class_counts = {label: int(samples_per_split * ratio)
                            for label, ratio in class_ratios.items()}
            self._fill_bucket(bucket, class_counts, group_ids, y)
            buckets.append(bucket)

        return buckets

    def _rotate_buckets(self, buckets):
        return buckets[-1:] + buckets[:-1]

    def _get_indices(self, bucket, group_ids, y):
        indices = []
        for label, groups in bucket.items():
            for group in groups:
                group_indices = np.where(group_ids == group)[0]
                label_indices = np.where(y == label)[0]
                indices.extend(np.intersect1d(group_indices, label_indices))
        return np.array(indices)

    def split(self, X, y, group_ids):
        index_map = np.arange(len(y))
        group_ids_s, y_s, index_map = shuffle(
            group_ids, y, index_map, random_state=self.random_state)

        class_ratios = {label: count / len(y)
                        for label, count in Counter(y).items()}
        buckets = self._create_buckets(group_ids_s, y_s, class_ratios)

        for _ in range(self.n_splits):
            train_buckets = buckets[1:]
            test_bucket = buckets[0]

            train_indices = np.concatenate(
                [np.array(bucket[label]) for bucket in train_buckets for label in bucket])
            test_indices = np.concatenate(
                [np.array(test_bucket[label]) for label in test_bucket])

            train_indices = index_map[train_indices]
            test_indices = index_map[test_indices]
            
            assert len(np.intersect1d(
                np.unique(group_ids.iloc[train_indices]), np.unique(group_ids.iloc[test_indices]))) == 0

            yield train_indices, test_indices
            buckets = self._rotate_buckets(buckets)


def create_folds(df, output_path, n_splits=5):
    files = df['label']
    labels = df['label']
    patient_ids = df['patient_id']

    # Divide whole dataset into train+val & test subsets ~80%-20% (stratified by label and patient_id)
    sgfk = StratifiedGroupKFold(n_splits=n_splits, random_state=42)
    train_val_index, test_index = next(sgfk.split(files, labels, patient_ids))

    df.iloc[test_index].to_csv(os.path.join(output_path, "test.csv"))

    train_val_files = df['label'].iloc[train_val_index]
    train_val_labels = df['label'].iloc[train_val_index]
    train_val_patient_ids = df['patient_id'].iloc[train_val_index]

    # Divide train+val into train & val subsets ~80%-20% (stratified by label and patient_id)
    sgfk = StratifiedGroupKFold(n_splits=n_splits, random_state=42)

    folds = sgfk.split(train_val_files, train_val_labels, train_val_patient_ids)
    
    print(f"Zoom: {zoom}x")
    print(f"output path: {output_path}\n")

    classes_balance_metadata = {}
    def check_balance(df, index, name):
        m_len = len(df.iloc[index][df.iloc[index].label == 1])
        b_len = len(df.iloc[index][df.iloc[index].label == 0])
        mp = len(df.iloc[index][df.iloc[index].label == 1]) / len(df.iloc[index])
        bp = len(df.iloc[index][df.iloc[index].label == 0]) / len(df.iloc[index])
        print(f"{name} - percent of B vs M samples: {bp:.2f} - {mp:.2f} / count: {b_len} : {m_len}")
        return {"percent_benign": bp, "percent_malignant": mp, "count_benign": b_len, "count_malignant": m_len}


    classes_balance_metadata["all"] = check_balance(df, range(len(df)), "All")
    classes_balance_metadata["test"] = check_balance(df, test_index, "Test")

    test_patient_ids = df.iloc[test_index].patient_id.unique()

    for idx, (train_index, val_index) in enumerate(folds):
        print("=============================================")
        print(f"Saving fold {idx}")
        train_index = map_nested_indices(train_index, train_val_index)
        val_index = map_nested_indices(val_index, train_val_index)

        # Check if there is no intersection between train/val/test datasets in terms of patient IDs
        train_patient_ids = df.iloc[train_index].patient_id.unique()
        val_patient_ids = df.iloc[val_index].patient_id.unique()
        assert intersection(train_patient_ids, val_patient_ids) == [] 
        assert intersection(train_patient_ids, test_patient_ids) == []
        assert intersection(val_patient_ids, test_patient_ids) == []
        
        classes_balance_metadata[f"train_fold_{idx}"] = check_balance(df, train_index, "Train")
        classes_balance_metadata[f"val_fold_{idx}"] = check_balance(df, val_index, "Val")

        df.iloc[train_index].to_csv(os.path.join(output_path, f"train_{idx}.csv"))
        df.iloc[val_index].to_csv(os.path.join(output_path, f"val_{idx}.csv"))

        with open(os.path.join(output_path, "classes_balance_metadata.json"), 'w') as metafile:
            json.dump(classes_balance_metadata, metafile, indent=4)

for dataset_name, dataset_content in datasets.items():
    print(f"Creating folds for {dataset_name}")
    create_folds(dataset_content["df"], dataset_content["path"])


