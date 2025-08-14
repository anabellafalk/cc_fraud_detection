import kagglehub
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

### Load and Split Data
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


### Extract Date Info
"""
Modifies dataframe `dat` creating the new column 'trans_year' containing the 
year of the transaction.
"""
def extract_year(dat):
    dat['trans_year'] = dat['trans_date_trans_time'].apply(lambda x: x[:4])

"""
Modifies dataframe `dat` creating the new column 'trans_month' containing the 
month of the transaction.
"""
def extract_month(dat):
    dat['trans_month'] = dat['trans_date_trans_time'].apply(lambda x: int(x[5:7]))

"""
Modifies dataframe `dat` creating the new column 'trans_month_year' containing the 
month and year of the transaction.
"""
def extract_month_year(dat):
    dat['trans_month_year'] = dat['trans_date_trans_time'].apply(lambda x: x[:7])


### Extract Age
"""
Modifies dataframe `dat` creating the new column 'age_at_trans' containing the 
age of credit card owner at time of transaction.
"""
def extract_age(dat):
    trans_date = pd.to_datetime(dat['trans_date_trans_time'].apply(lambda x: x[:10]))
    dob = pd.to_datetime(dat['dob'])
    dat['age_at_trans'] = (trans_date - dob).dt.days // 365


### Bin Jobs
"""
FIt KMeans with 3 clusters to the 'job' column to get bins. 
Returns fitted cluster to create mappings
"""
def fit_job_bins(dat):
    cluster = KMeans(3)
    fraud_job = dat.groupby('job')['is_fraud'].mean()
    cluster.fit(fraud_job.values.reshape(-1,1))
    return cluster

"""
Modifies dataframe `dat` adding the binned column 'job_bin' for the column 'job'
using fitted `cluster` to create binning mapping
"""
def bin_jobs(dat, cluster):
    fraud_job = dat.groupby('job')['is_fraud'].mean()
    bin_map = pd.Series(cluster.predict(fraud_job.values.reshape(-1,1)))
    bin_map.index = fraud_job.index
    dat['job_bin'] = dat['job'].apply(lambda x: bin_map.loc[x])


### Extract Distance
"""
Modifies dataframe `dat` creating the new column 'distance' containing the manhattan
distance from the owner's address to the merchant.
"""
def extract_distance(dat):
    dat['distance'] = np.sqrt((dat['merch_lat'] - dat['lat'])**2 + (dat['merch_long'] - dat['long'])**2)


### Scale Numeric Variables
"""
Fit standard scaler to numeric column
Returns scaler
"""
def fit_col_scaler(dat, col):
    scaler = StandardScaler()
    scaler.fit(dat[col].values.reshape(-1,1))
    return scaler

"""
Modifies dataframe `dat` creating the new column '`col`_scale' containing the 
standard scaled version of the values in `col`
"""
def scale_col(dat, col, scaler):
    dat[col + '_scale'] = scaler.transform(dat[col].values.reshape(-1,1))


### Bin Numeric Variables
def fit_age_bins(dat):
    dat['age_bins'] = pd.qcut(dat['age_at_trans'].values, 7)
    return dat['age_bins'].unique()

### Encode Categorical Variables
"""
Fit OneHotEncoder to categorical column
Returns encoder
"""
def fit_col_encoder(dat, col):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(dat[col].values.reshape(-1,1))
    return encoder

"""
One Hot Encodes the categorical variable `col` in `dat`. 
Returns a dataframe of the encoded categories. Each column name begins with 
`col` followed by the category name after a '.'
"""
def encode_col(dat, col, encoder):
    encoded = pd.DataFrame(encoder.transform(dat[col].values.reshape(-1,1)))
    encoded.columns = col + "." + encoder.categories_[0]
    return encoded

