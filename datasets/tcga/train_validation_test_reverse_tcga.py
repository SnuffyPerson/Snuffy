import os
import shutil

# TODO This code has not been tested!

# define the main folder
main_folder = 'single'

# Define the fold folder
fold_number = '1'
fold_folder = f"single/fold{fold_number}"

# Define the subfolders
split_folders = ['train', 'validation', 'test']

# Define the target folders
class_folders = ['0_luad', '1_lusc']

# Iterate over the main folder and subfolders
for split_folder in split_folders:
    subfolder_path = os.path.join(fold_folder, split_folder)

    # Iterate over the target folders
    for class_folder in class_folders:
        target_folder_path = os.path.join(fold_folder, split_folder, class_folder)
        for wsi in os.listdir(target_folder_path):
            wsi_path = os.path.join(target_folder_path, wsi)
            print('wsi_path:', wsi_path)
            print('main_folder:', main_folder)
            print('target_folder:', class_folder)
            print(f'{main_folder}/{class_folder}')
            shutil.move(wsi_path, f'{main_folder}/{class_folder}')
