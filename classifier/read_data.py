import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import PATH_CONSTANTS as PATHS

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

base_path = PATHS.DATASET_LOCATION
NORMALIZE_VALUE = 1300  # Max value coordinates can take based on size of image. Adjust as necessary
TOTAL_MOVES = 17  # Number of different exercises in dataset


def get_pars():
    return os.listdir(base_path)


def get_header():
    par = os.listdir(base_path)[0]
    par_path = base_path + par + '\\'

    move = os.listdir(par_path)[0]
    move_path = par_path + str(move) + '\\'

    csv = os.listdir(move_path)[0]
    csv_path = move_path + csv

    data = pd.read_csv(csv_path, header=0)
    return list(data.columns.values)


def get_dataset(leave_out=[], withhold_moves=[], par_count=None, val_percent=0.1, test_percent=0.1, normalize=False, one_hot=False, flatten=False, verbose=True):
    """
    Reads data from disc and processes it according to parameters to provide data in the correct ways for XGBC and CNN models

    @param leave_out: list of participants that will be left out of data set
    @param withhold_moves: list of moves that will be left out of data set
    @param par_count: number of participants to include in dataset. Set to None to include all participants
    @param val_percent: number between 0 and 1 indicating proportion of data used to create validation set
    @param test_percent: number between 0 and 1 indicating proportion of data used to create test set
    @param normalize: boolean indicating whether normalization is performed or not
    @param one_hot: boolean indicating whether labels are one-hot encoded or not
    @param flatten: boolean indicating whether input matrices are flattened or not
    @param verbose: boolean turning progress text on or off
    @return: list containing train, validation and test data
    """
    dataset_list = []
    test_list = []
    withhold_list = []

    pars = os.listdir(base_path)
    if par_count:
        pars = pars[:par_count]

    for par in pars:
        par_path = base_path + par + '\\'

        if verbose:
            print('Reading', par)

        moves = os.listdir(par_path)
        for move in moves:
            move_path = par_path + str(move) + '\\'

            csvs = os.listdir(move_path)
            for csv in csvs:
                csv_path = move_path + csv

                pose_train = pd.read_csv(csv_path, header=0)

                pose_features = np.array(pose_train)
                if normalize:
                    pose_features = pose_features/NORMALIZE_VALUE
                else:
                    pose_features = pose_features.astype(int)

                label = int(move)
                if one_hot:
                    one_hot = [0 for i in range(len(moves))]
                    one_hot[int(move)-1] = 1
                    label = np.array(one_hot)

                first = [pose_features[:100], label]
                last = [pose_features[-100:], label]
                length = len(pose_features)
                mid_start = (length // 2) - 50
                mid = [pose_features[mid_start:mid_start + 100], label]

                cuts = [first, last, mid]
                cuts_array = [np.array(x, dtype=object) for x in cuts]

                # If we're doing leave one out, add that participant to separate list
                if par in leave_out:
                    test_list.extend(cuts_array)
                elif int(move) in withhold_moves:
                    withhold_list.extend(cuts_array)
                else:
                    dataset_list.extend(cuts_array)

    dataset_list = np.asarray(dataset_list, dtype=object)
    test_list = np.asarray(test_list, dtype=object)
    withhold_list = np.asarray(withhold_list, dtype=object)

    x_train, y_train = zip(*dataset_list)

    if leave_out:
        x_test, y_test = zip(*test_list)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_percent)

        if len(withhold_moves) > 0:
            wx, wy = zip(*withhold_list)
            x_dispose, x_extra, y_dispose, y_extra = train_test_split(wx, wy, test_size=test_percent)
            np.concatenate(x_test, x_extra)
            np.concatenate(y_test, y_extra)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_percent)

    if flatten:
        x_train = [a.flatten() for a in x_train]
        x_val = [a.flatten() for a in x_val]
        x_test = [a.flatten() for a in x_test]

    return [np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)]
