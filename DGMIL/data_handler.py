from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

PATH_TO_DATASETS = 'datasets/Camelyon16/'
PATH_TO_SAVE = './'
BENIGN_CSV = '0-normal.csv'
MALIGNANT_CSV = '1-tumor.csv'


class CustomDataSet(Dataset):
    """
    class to read data from csv files in the hard disk
    """

    def __init__(self, path_df, idx):
        self.path_df = path_df
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, x):
        idx = self.idx[x]
        path = self.path_df.iloc[idx, 0]
        return pd.read_csv(path).to_numpy()


def get_idx(df, frac):
    """
    shuffles a list of indexes, then chooses a fraction of them for train and the rest for test
    """
    idx = df.index.to_numpy()
    np.random.shuffle(idx)
    return idx[:int(len(idx) * frac)], idx[int(len(idx) * frac):]


def get_csv_numpy_array(df, index, max_rows=-1):
    """
    df: DataFrame containing path to WSI files
    index: index of file to be chosen
    max_rows: max number of files to be read, set to -1 to read all indexes
    retrieves all the slides of given indexes
    """
    path_list = df.loc[index, '0'][:max_rows].to_numpy()
    return pd.concat(map(pd.read_csv, path_list)).to_numpy()


def get_num_bag_list_index(df, index, max_rows=-1):
    """
    generate list index of start and end rows of test WSI's in a matrix where each row corresponds to a patch
     and n consecutive rows can make a WSI.
    """
    num_bag_list_index = np.array([0] + [pd.read_csv(i).to_numpy().shape[0] for i in df.loc[index, '0'][:max_rows]])
    for i in range(1, len(num_bag_list_index)):
      num_bag_list_index[i] += num_bag_list_index[i - 1]
    return num_bag_list_index


def generate_data(train_frac=0.95, batch_size=10, shuffle=False):
    # read the DataFrames containing paths to embeddings and label
    df_neg = pd.read_csv(PATH_TO_DATASETS + MALIGNANT_CSV)
    df_pos = pd.read_csv(PATH_TO_DATASETS + BENIGN_CSV)
    # split negative and positive WSI's into train and test
    neg_train_idx, neg_test_idx = get_idx(df_neg, train_frac)
    pos_train_idx, pos_test_idx = get_idx(df_pos, train_frac)

    # read positive test WSI's, concat them and save them
    test_pos = get_csv_numpy_array(df_pos, pos_test_idx)
    np.save(PATH_TO_SAVE + 'MAE_testing_pos_feats.npy', test_pos)
    # read negative test WSI's, concat them and save them
    test_neg = get_csv_numpy_array(df_neg, neg_test_idx)
    np.save(PATH_TO_SAVE + 'MAE_testing_neg_feats.npy', test_neg)
    # save all test WSI's
    np.save(PATH_TO_SAVE + 'test_MAE_feats.npy', np.concatenate((test_neg, test_pos)))
    del test_neg, test_pos
    # generate numb_bag_list_index for positive and negative slides and savbe them
    num_bag_list_index_pos = get_num_bag_list_index(df_pos, pos_test_idx)
    num_bag_list_index_neg = get_num_bag_list_index(df_neg, neg_test_idx)
    np.save(PATH_TO_SAVE + 'num_bag_list_index.npy',
            np.concatenate((num_bag_list_index_neg, num_bag_list_index_pos[1:] + num_bag_list_index_neg[-1])))
    del num_bag_list_index_pos, num_bag_list_index_neg
    # make label for positive and negative slides and savbe them
    labels = np.array(['n'] * len(neg_test_idx) + ['p'] * len(pos_test_idx))
    np.save(PATH_TO_SAVE + 'test_slide_label.npy', labels)

    negative_data_set = CustomDataSet(df_neg, neg_train_idx)
    positive_data_set = CustomDataSet(df_pos, pos_train_idx)

    # make DataLoaders of negative and positive WSI's which will be working in parallel
    negative_data_loader = DataLoader(negative_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=np.concatenate)
    positive_data_loader = DataLoader(positive_data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=np.concatenate)

    return negative_data_loader, positive_data_loader
