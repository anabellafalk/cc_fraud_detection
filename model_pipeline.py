import data_prep as dp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

"""
Prepares data `dat` to be used as modeling input.
Assumes `dat` is in same format as data model trained on
"""
def prepare_data(dat, train=False):
    # Load train data
    dat_tr = dp.load_train()

    if train:
        dat=dat_tr

    # Extract age
    dp.extract_age(dat)

    # Extract distance
    dp.extract_distance(dat)

    # Encode category
    category_encoder = dp.fit_col_encoder(dat_tr, 'category')
    category_enc = dp.encode_col(dat, 'category', category_encoder)

    # Extract year
    dp.extract_year(dat)

    # Encode year
    year_encoder = dp.fit_col_encoder(dat_tr, 'trans_year')
    year_enc = dp.encode_col(dat, 'trans_year', year_encoder)

    # Extract month
    dp.extract_month(dat)
    dat['trans_month'] = dat['trans_month'].astype(str)
    month_encoder = dp.fit_col_encoder(dat_tr, 'trans_month')
    month_enc = dp.encode_col(dat, 'trans_month', month_encoder)

    # Bin jobs
    job_binner = dp.fit_job_bins(dat_tr)
    dp.bin_jobs(dat_tr, job_binner)
    dp.bin_jobs(dat, job_binner)
    dat_tr['job_bin'] = dat_tr['job_bin'].astype(str)
    dat['job_bin'] = dat['job_bin'].astype(str)
    # Encode jobs
    job_bin_encoder = dp.fit_col_encoder(dat_tr, 'job_bin')
    job_bin_enc = dp.encode_col(dat, 'job_bin', job_bin_encoder)

    model_dat = pd.concat([
    dat[['amt'] + ['age_at_trans', 'city_pop', 'distance']].reset_index(drop=True),
    category_enc.reset_index(drop=True),
    year_enc.reset_index(drop=True),
    month_enc.reset_index(drop=True),
    job_bin_enc.reset_index(drop=True)
    ], axis=1)

    if train:
        model_dat = pd.concat([
        dat[['is_fraud'] + ['amt'] + ['age_at_trans', 'city_pop', 'distance']].reset_index(drop=True),
        category_enc.reset_index(drop=True),
        year_enc.reset_index(drop=True),
        month_enc.reset_index(drop=True),
        job_bin_enc.reset_index(drop=True)
        ], axis=1)

    return model_dat

"""
Prepares training data to fit the model
"""
def prepare_train_data():
    return prepare_data(None, True)
    
"""
Fits the optimal random forest model on prepared data `dat`.
Returns fitted model object
"""
def fit_model(dat):
    rfc = RandomForestClassifier(class_weight='balanced', min_samples_leaf=9, min_samples_split=2, n_estimators=500)
    rfc.fit(dat.drop('is_fraud', axis=1), dat['is_fraud'])

    return rfc

"""
Returns `model` predictions on prepared data `dat`
"""
def predict(model, dat):
    return model.predict(dat.drop('is_fraud', axis=1))

"""
Saves `model` object with `name` to a pkl file
"""
def save_model(model, name):
    joblib.dump(model, name+'.pkl')

"""
Fits then saves the model with file `name`
"""
def fit_save_model(name):
    dat = prepare_train_data()
    model = fit_model(dat)
    save_model(model, name)

