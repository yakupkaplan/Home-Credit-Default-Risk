# HELPER FUNCTIONS FOR HOME CREDIT DEFAULT PREDICTION

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    """
    One-hot encoding for categorical columns with get_dummies
    Returns dataframe with one-hot encoded columns and list for new created columns

    :param df: dataframe
        dataframe whose categorical columns will be one hot encoded

    :param nan_as_category: bool
        boolean indicating if the missing values will be shown separately or not.

    :return: dataframe, list for new_columns

    """
    import pandas as pd
    import re
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Rare encoding function for rare labels
def rare_encoder(dataframe, rare_perc):
    """
    Rare encoding function for rare labels
    Returns dataframe with 'Rare' encoded labels

    :param dataframe: dataframe
        dataframe to be rare encoded

    :param rare_perc: float
        percentage for lables to be accepted as 'Rare'

    :return: dataframe

    """
    import numpy as np
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df


# command line access for debuging
def get_namespace():
    """
    Command line access for debuging.
    Returns parser argument indicates boolean True or False

    :return: bool

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)
    return parser.parse_args()


# Display/plot feature importance
def display_importances(feature_importance_df_):
    """
    Displays/plots and saves feature importances.

    :param feature_importance_df_: dataframe
        dataframe for feature importances to be plotted and saved.

    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


# # saving models
# def saving_models():
#     """
#     Saving models for later use
#
#     """
#     import os
#     import pickle
#     cur_dir = os.getcwd()
#     os.chdir('/models/reference/')
#     model_name = "lightgbm_fold_" + str(n_fold + 1) + "." + "pkl"
#     pickle.dump(model, open(model_name, 'wb'))  # model
#     os.chdir(cur_dir)