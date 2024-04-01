import numpy as np
import pandas as pd
import os

PATH_TO_DATASETS = 'datasets/Camelyon16/'
PATH_TO_SAVE = './'
BENIGN_CSV = '0-normal.csv'
MALIGNANT_CSV = '1-tumor.csv'

u = -1
trn = 4 * u
tst = 1 * u

print(f'total number of slides: {2 * (trn + tst)}')


def get_csv_numpy_array(df, index, max_rows=-1):
    path_list = df.loc[index, '0'].iloc[:max_rows].to_numpy()
    return pd.concat(map(lambda x: pd.read_csv(x, dtype=np.float32), path_list)).to_numpy()


def get_num_bag_list_index(df, index, max_rows=-1):
    num_bag_list_index = np.array([0] + [pd.read_csv(i).to_numpy().shape[0] for i in df.loc[index, '0'].iloc[:max_rows]])
    for i in range(1, len(num_bag_list_index)):
      num_bag_list_index[i] += num_bag_list_index[i - 1]
    return num_bag_list_index


def get_index(df, frac_test=0.2):
    # index of test and train slides
    idx = df.index.to_numpy()
    np.random.shuffle(idx)
    train_idx = idx[int(np.ceil(frac_test * idx.shape[0])):]
    test_idx = idx[:int(np.ceil(frac_test * idx.shape[0]))]
    # train_idx = idx[:int(np.ceil(frac_test * idx.shape[0]))]
    # test_idx = idx[-int(np.ceil(frac_test * idx.shape[0])):]
    return train_idx, test_idx


if not os.path.exists(PATH_TO_SAVE):
    os.makedirs(PATH_TO_SAVE)

df_neg = pd.read_csv(PATH_TO_DATASETS + MALIGNANT_CSV)
index_train_neg, index_test_neg = get_index(df_neg)

print('generatring train negative features')
np.save(PATH_TO_SAVE + 'MAE_dynamic_trainingneg_feats.npy', get_csv_numpy_array(df_neg, index_train_neg, max_rows=trn))
print('done')
del index_train_neg

df_pos = pd.read_csv(PATH_TO_DATASETS + BENIGN_CSV)
index_train_pos, index_test_pos = get_index(df_pos)

print('generatring train positive features')
np.save(PATH_TO_SAVE + 'MAE_dynamic_trainingpos_feats', get_csv_numpy_array(df_pos, index_train_pos, max_rows=trn))
print('done')
del index_train_pos

print('generatring test positive features')
test_pos = get_csv_numpy_array(df_pos, index_test_pos, max_rows=tst)
np.save(PATH_TO_SAVE + 'MAE_testing_pos_feats.npy', test_pos)
print('done')

print('generatring test negative features')
test_neg = get_csv_numpy_array(df_neg, index_test_neg, max_rows=tst)
np.save(PATH_TO_SAVE + 'MAE_testing_neg_feats.npy', test_neg)
print('done')

print('generatring all test features')
np.save(PATH_TO_SAVE + 'test_MAE_feats.npy', np.concatenate((test_neg, test_pos)))
print('done')
del test_neg, test_pos

print('generatring num_bag_list_index')
num_bag_list_index_pos = get_num_bag_list_index(df_pos, index_test_pos, max_rows=tst)
num_bag_list_index_neg = get_num_bag_list_index(df_neg, index_test_neg, max_rows=tst)
np.save(PATH_TO_SAVE + 'num_bag_list_index.npy',
        np.concatenate((num_bag_list_index_neg, num_bag_list_index_pos[1:] + num_bag_list_index_neg[-1])))
print('done')
del num_bag_list_index_pos, num_bag_list_index_neg

print('generatring test_slide_label')
labels = np.array(['n'] * len(index_test_neg[:tst]) + ['p'] * len(index_test_pos[:tst]))
np.save(PATH_TO_SAVE + 'test_slide_label.npy', labels)
print('done')
del labels
