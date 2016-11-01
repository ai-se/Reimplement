from __future__ import division
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import random


def add_feature_weights(filename, sampling_percent=10):
    name = filename.split('/')[-1]
    content = pd.read_csv(filename)
    size_content = content.shape[0]
    sampling_rows = int(size_content * sampling_percent/100)
    rows_no = random.sample(content.index, sampling_rows)
    rows = content.ix[rows_no]

    indep_col = [c for c in content.columns if "$<" not in c]
    dep_col = [c for c in content.columns if "$<" in c]
    model = DecisionTreeRegressor()
    model.fit(rows[indep_col], rows[dep_col])
    importance = [c*100 for c in list(model.feature_importances_)]

    indep_content = content[indep_col]
    dep_content = content[dep_col]

    for index,col in enumerate(indep_content.columns):
        col_f = "##" + col
        indep_content[col_f] = (indep_content[col] * importance[index])

    norm_cols = [c for c in indep_content.columns if "##" in c]
    indep_f = indep_content[norm_cols]
    indep_f.columns = indep_col
    new_df = pd.concat([indep_f, dep_content], axis=1)
    new_df.to_csv("./FData/norm_" + name, index=False)
    return "./FData/norm_" + name
