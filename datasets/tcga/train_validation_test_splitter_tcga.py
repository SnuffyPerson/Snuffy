import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import KFold

# TODO For TCGA we should make sure that Different patients aren’t in different parts of split -> Not handled yet.

"""
Train/Valid/Test Ratio: 0.60/0.15/0.25
    As the K in KFold is set to 4, the test split is 0.25. 
    As the test_size (line 49) is set to 0.2, the train/valid will be 0.6/0.15.
"""


def create_reference_csv():
    paths = ['single/0_luad', 'single/1_lusc']
    slide_names = []

    for path in paths:
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                slide_names.append(name)

    df = pd.DataFrame(slide_names, columns=['slide'])
    df = df.sort_values(by=['slide'])
    df.to_csv('danial_generated_reference.csv', index=False)


create_reference_csv()

# Step 1: Load the data from reference.csv
data = pd.read_csv("danial_generated_reference.csv")

# Step 2: Extract values from the "image" column
image_list = data["slide"].tolist()

# step 3: 5 Fold
random.seed(42)  # Set the random seed for reproducibility
random.shuffle(image_list)
num_folds = 4
kf = KFold(n_splits=num_folds)
fold = 1
for train_index, test_index in kf.split(image_list):
    if fold > 1:
        break
    train_validation_images = [image_list[i] for i in train_index]
    test_images = [image_list[i] for i in test_index]
    fold += 1

# Step 4: Split the list into train, validation, and test sets
train_images, validation_images = train_test_split(train_validation_images, test_size=0.2, random_state=42)

# Step 5: Create train, validation, and test folders
base_dir = "single"
fold_number = '1'
train_dir = os.path.join(base_dir, f'fold{fold_number}', "train")
validation_dir = os.path.join(base_dir, f'fold{fold_number}', "validation")
test_dir = os.path.join(base_dir, f'fold{fold_number}', "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 6: Organize folders based on train_images, validation_images, and test_images
for folder in ["0_luad", "1_lusc"]:
    for image in train_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(train_dir, folder, image)
            shutil.move(src, dst)

    for image in validation_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(validation_dir, folder, image)
            shutil.move(src, dst)

    for image in test_images:
        src = os.path.join(base_dir, folder, image)
        if os.path.exists(src):
            dst = os.path.join(test_dir, folder, image)
            shutil.move(src, dst)
