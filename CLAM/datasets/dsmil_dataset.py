from typing import Tuple, Optional
import os, time
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF

from .dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_Split

EMBEDDINGS_PATH = '../research/Research/dsmil-wsi/embeddings/'
FEATS_PATH = '../research/Research/dsmil-wsi/'
SAVE_PATH = 'runs/'

SHUFFLE_SEED = 42


def get_bag_feats(csv_file_df, args):
    feats_csv_path = csv_file_df.iloc[0]
    feats_csv_path = feats_csv_path.replace("datasets/Camelyon16/",
                                            "embeddings/camelyon16/official/")  # Only for official feats
    df = pd.read_csv(os.path.join(FEATS_PATH, feats_csv_path))

    feat_labels_available = 'position' in df and 'label' in df

    # get bag feats
    feats = shuffle(df, random_state=SHUFFLE_SEED).reset_index(drop=True)
    if feat_labels_available:
        feats = feats.drop(columns=['label', 'position'], inplace=False)
    feats = feats.to_numpy()

    # get bag label
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1]) <= (len(label) - 1):
            label[int(csv_file_df.iloc[1])] = 1

    # get bag feats labels and their positions (if available)
    positions = None
    feats_labels = None
    if feat_labels_available:
        feats_labels = df['label'].to_numpy()  # +
        positions = list(df['position'])

    return label, feats, feats_labels, positions


def _load_data(data_frame, args):
    all_labels = []
    all_feats = []
    all_feats_labels = []
    all_positions = []

    feats_labels_available = True

    for i in tqdm(range(len(data_frame))):
        label, feats, feats_labels, positions = get_bag_feats(data_frame.iloc[i], args)
        all_labels.append(label)
        all_feats.append(feats)

        if feats_labels is None:
            feats_labels_available = False
        if feats_labels_available:
            all_feats_labels.append(feats_labels)
            all_positions.append(positions)

    if not feats_labels_available:
        all_feats_labels = None
        all_positions = None

    return all_labels, all_feats, all_feats_labels, all_positions


def _load_data_mp_worker(args):
    i, row, args = args
    label, feats, feats_labels, positions = get_bag_feats(row, args)
    return label, feats, feats_labels, positions


def _load_data_mp(data_frame, args):
    with mp.Pool(processes=args.num_processes) as pool:
        all_labels, all_feats, all_feats_labels, all_positions = zip(
            *pool.map(_load_data_mp_worker, [(i, data_frame.iloc[i], args) for i in range(len(data_frame))])
            # *pool.map(_load_data_mp_worker, [(i, data_frame.iloc[i], args) for i in range(5)])
        )
    all_labels, all_feats, all_feats_labels, all_positions = (
        list(all_labels), list(all_feats), list(all_feats_labels), list(all_positions)
    )

    feats_labels_available = all_feats_labels[0] is not None
    if not feats_labels_available:
        all_feats_labels = None
        all_positions = None

    return all_labels, all_feats, all_feats_labels, all_positions


def load_data(dataframe, args, use_mp=False):
    if use_mp:
        return _load_data_mp(dataframe, args)
    else:
        return _load_data(dataframe, args)


class DSMILDataPreloader:
    class args:
        dataset = 'camelyon16'
        embedding = 'official'
        num_processes = 8
        num_classes = 1
        split = 0.2

    def __init__(self, split_name):
        self.split_name = split_name
        self.all_labels, self.all_feats, self.all_feats_labels, self.all_positions = self._get_official_data()

    def _get_official_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray], Optional[NDArray]]:
        bags_csv = os.path.join(EMBEDDINGS_PATH, self.args.dataset, 'official', self.args.dataset.capitalize() + '.csv')
        bags_path = pd.read_csv(bags_csv)
        split_path = self._get_split_by_args(bags_path)
        split_path = shuffle(split_path, random_state=SHUFFLE_SEED).reset_index(drop=True)
        split_data = self._load_split_data(split_path, self.split_name)

        return split_data

    def _get_split_by_args(self, bags_path):
        train_path = bags_path.iloc[0:int(len(bags_path) * (1 - self.args.split)), :]
        valid_path = bags_path.iloc[int(len(bags_path) * (1 - self.args.split)):, :]
        valid_path, test_path = (
            valid_path.iloc[0:len(valid_path) // 2, :],
            valid_path.iloc[len(valid_path) // 2:, :]
        )
        translation = {
            'train': train_path,
            'valid': valid_path,
            'test': test_path
        }
        split_path = translation.get(self.split_name)
        return split_path

    def _load_split_data(self, split_path, split_name):
        use_mp = True
        print(f'Loading {split_name} data... (mp={use_mp})...')
        start_time = time.time()
        data = load_data(split_path, self.args, use_mp=use_mp)
        print(f'DONE (Took {(time.time() - start_time):.1f}s)')
        return data


class DSMILDataset:
    def __init__(self, split_name=None):
        self.slide_data = self._create_slide_data_csv()
        if split_name is not None:
            self.data: DSMILDataPreloader = DSMILDataPreloader(split_name)

    def return_splits(self, from_id, csv_path):
        return DSMILDataset('train'), DSMILDataset('valid'), DSMILDataset('test')

    def _create_slide_data_csv(self):
        # raise NotImplementedError
        bags_csv = os.path.join(EMBEDDINGS_PATH, 'camelyon16', 'official', 'Camelyon16' + '.csv')
        bags_path = pd.read_csv(bags_csv)
        num_slides = len(bags_path)
        all_labels = bags_path['label']
        bags_path['case_id'] = ['patient_0' for _ in range(num_slides)]
        bags_path['slide_id'] = [f'slide_{i}' for i in range(num_slides)]
        bags_path['label'] = ['tumor_tissue' if label == 1 else 'normal_tissue' for label in all_labels]
        bags_path.to_csv('Camelyon16_adapted.csv')
        return bags_path

    def __len__(self):
        return len(self.data.all_labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data.all_feats[idx]).float(), self.data.all_labels[idx].tolist()[0]
