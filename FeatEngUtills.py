# FUNCTIONS FEATURE ENGINEERING FOR HOME CREDIT PROJECT

import pandas as pd
import numpy as np
import pymysql as pymysql
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning


import warnings

warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Function to evaluate missing values in dataset
def missing_values_table(df):
    """
    Function to evaluate missing values in dataset
    :param df: dataframe for examining missing values
    :return: mis_val_table_ren_columns
    """
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    """
    Function to calculate outlier thresholds

    :param dataframe: dataframe
    :param variable: variable name to define thresholds
    :return: low_limit and up_limit for variable to be outlier
    """
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    """
    Function to report variables with outliers and return the names of the variables with outliers with a list

    :param dataframe: dataframe
    :param num_col_names: list for numerical variables
    :param plot: boolean for plotting boxplot
    :return: variable_names with outliers
    """
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    """
    Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method

    :param dataframe: dataframe
    :param variable: variable name
    :return: dataframe with reassigned values
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Define a function to apply one hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    """
    Function to apply one hot encoding to categorical variables.

    :param dataframe: dataframe to be one hot encoded.
    :param categorical_cols: list for categorical columns to be one hot encoded
    :param nan_as_category: Boolean for creating extra column for missing values or not
    :return: dataframe with one hot encoded columns, list for new_columns after one hot encoding
    """
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


# Function to label age ranges
def get_age_label(days_birth):
    """
    Function to label age ranges
    :param days_birth:
    :return: the age group label (int).
    """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0


# Function to label working years ranges
def get_working_year_label(days_employed):
    """
    Function to label working years ranges
    :param days_employed:
    :return: the working year group label (int).
    """
    working_years = -days_employed / 365
    if working_years < 3: return 1
    elif working_years < 5: return 2
    elif working_years < 10: return 3
    elif working_years < 15: return 4
    elif working_years < 20: return 5
    else: return 6