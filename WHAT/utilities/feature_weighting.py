from __future__ import division
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import random


def add_feature_weights_random(filename, sampling_percent=10):
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
    # print importance

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


def add_feature_weights(filename, sampling_percent=10):
    def check_condition(feature_table, threshold=5):
        for feature in feature_table:
            for key in feature.keys():
                if feature[key] < threshold: return False
        return True

    def get_samples(content):
        freq_table = [{} for _ in xrange(len(content.columns)-1)]
        for i, col in enumerate(content.columns):
            if "$<" not in col:
                distinct_values = content[col].unique()
                for distinct_value in distinct_values:
                    freq_table[i][distinct_value] = 0
        indep_col = [c for c in content.columns if "$<" not in c]

        chosen_indices = []
        remaining_indices = range(content.shape[0])

        while True:
            chosen_index = random.randint(0, len(remaining_indices)-1)
            chosen_config = content.iloc[remaining_indices[chosen_index]]

            # update the freq_table
            for iicol, icol in enumerate(indep_col):
                freq_table[iicol][chosen_config[icol]] += 1

            chosen_indices.append(remaining_indices[chosen_index])
            # remove from remaining_indices = []
            remaining_indices.pop(chosen_index)

            assert(len(chosen_indices) + len(remaining_indices) == len(content)), "Something is wrong"
            if check_condition(freq_table) is True: break
        return content.ix[chosen_indices]

    name = filename.split('/')[-1]
    content = pd.read_csv(filename)
    rows = get_samples(content)

    indep_col = [c for c in content.columns if "$<" not in c]
    dep_col = [c for c in content.columns if "$<" in c]
    model = DecisionTreeRegressor()
    model.fit(rows[indep_col], rows[dep_col])
    importance = [c*100 for c in list(model.feature_importances_)]
    # print importance

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