from __future__ import division
import pandas as pd
from sklearn import preprocessing


def do_normalize_min_max(filename):
    name = filename.split('/')[-1]
    df = pd.read_csv(filename)
    columns = df.columns
    indep = [c for c in columns if "$<" not in c]
    dep = [c for c in columns if "$<" in c]
    indep_values = df[indep]
    dep_values = df[dep]

    x = indep_values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    indep_norm = pd.DataFrame(x_scaled, columns=indep)

    new_df = pd.concat([indep_norm, dep_values], axis=1)
    new_df.to_csv("./NData/MinMax/norm_" + name, index=False)
    return "./NData/MinMax/norm_" + name


def do_normalize_zscore(filename):
    name = filename.split('/')[-1]
    df = pd.read_csv(filename)
    columns = df.columns
    indep = [c for c in columns if "$<" not in c]
    dep = [c for c in columns if "$<" in c]
    indep_values = df[indep]
    dep_values = df[dep]

    for col in indep_values.columns:
        col_norm = '##' + col
        if indep_values[col].std(ddof=0) != 0:
            indep_values[col_norm] = (indep_values[col] - indep_values[col].mean())/indep_values[col].std(ddof=0)
        else:
            # useless columns
            indep_values[col_norm] = indep_values[col]

    norm_cols = [c for c in indep_values.columns if "##" in c]
    indep_norm = indep_values[norm_cols]
    indep_norm.columns = indep
    new_df = pd.concat([indep_norm, dep_values], axis=1)
    new_df.to_csv("./NData/zscore/norm_" + name, index=False)
    return "./NData/zscore/norm_" + name