import kagglehub
import numpy as np
import pandas as pd

"""
Load training data from kaggle. 
Returns pandas df with "Unnamed: 0" column dropped.
"""
def load_train():
    # Download latest version
    path = kagglehub.dataset_download("kartik2112/fraud-detection")

    print("Path to dataset files:", path)

    file = "/fraudTrain.csv"
    dat = pd.read_csv(path + file)

    dat = dat.drop('Unnamed: 0', axis = 1)

    return dat

"""
Split data into training and testing based on observed split.
Appx 25% validation and 75% training.
Returns two pandas df each of training and validation data.
"""
def split_train_val(dat):
    n_cc = dat["cc_num"].nunique()

    # Randomly sample cc #s for validation
    np.random.seed(42)  # set seed
    unique_cc = dat["cc_num"].unique()
    cc_val = np.random.choice(unique_cc, size = int(0.3 * n_cc))

    # Split data
    dat_val = dat[dat["cc_num"].isin(cc_val)]
    dat_tr = dat[~dat["cc_num"].isin(cc_val)]

    return dat_tr, dat_val


"""
Loads training data from kaggle then splits into training and validation.
Returns two pandas df each of training and validation data.
"""
def load_train_split():
    return split_train_val(load_train())
