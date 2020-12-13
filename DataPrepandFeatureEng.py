# HOME CREDIT DEFAULT RISK PROJECT

# DATA PREPROCESSING AND FEATURE ENGINEERING

"""
In this project we try to predict home credit default risk for clients.
In this script we focus on data preprocessing and feature engineering.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/overview

Steps to follow for data preprocessing and feature engineering:
    - New Features Creation and Analysis of New Features
    - Missing Values Analysis, but not treatment
    - Outlier Analysis, but not treatment
    - Label and One Hot Encoding
    - Standardization / Feature Scaling
    - Control the Dataset
    - Save Dataset for Modeling

"""


# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pickle

from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Import functions from FeatEngUtills.py.
import FeatEngUtills as feateng

# Check directories
import os
os.getcwd()

# Load datasets
application_train = pd.read_csv('datasets/application_train.csv')
application_test = pd.read_csv('datasets/application_test.csv')
# bureau = pd.read_csv('datasets/bureau.csv')
# bureau_balance = pd.read_csv('datasets/bureau_balance.csv')
# credit_card_balance = pd.read_csv('datasets/credit_card_balance.csv')
# installments_payments = pd.read_csv('datasets/installments_payments.csv')
# previous_application = pd.read_csv('datasets/previous_application.csv')

# Join train and test tables
df = application_train.append(application_test)
df.head()

# Catch categorical and numerical variables and save in a list
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ['SK_ID_CURR', 'TARGET']]


## FEATURE ENGINEERING

# 'DAYS_EMPLOYED'. There are some strange values. These will be replaced with np.nan.
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
# New feature to show the ratio 'DAYS_EMPLOYED' / 'DAYS_BIRTH'.
df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
# Show results for new feature
df['NEW_DAYS_EMPLOYED_PERC'].head()
df['NEW_DAYS_EMPLOYED_PERC'].describe()
df.groupby('TARGET').agg({'NEW_DAYS_EMPLOYED_PERC': [min, max, np.mean, np.median]})


# Remove super outliers from the dataframe for AMT_INCOME_TOTAL
df = df[df['AMT_INCOME_TOTAL'] < 20000000]
# Feauture to show the ratio 'AMT_INCOME_TOTAL' / 'AMT_CREDIT'
df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'] # ? Comparison by year *12, meaningful or not for tree based models. Canceled.
# Show results for new feature
df['NEW_INCOME_CREDIT_PERC'].head()
df['NEW_INCOME_CREDIT_PERC'].describe()
df.groupby('TARGET').agg({'NEW_INCOME_CREDIT_PERC': ['count', min, max, np.mean, np.median]})


# Feauture to show the ratio 'AMT_INCOME_TOTAL' / 'CNT_FAM_MEMBERS'
df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
# Show results for new feature
df['NEW_INCOME_PER_PERSON'].head()
df['NEW_INCOME_PER_PERSON'].describe()
df.groupby('TARGET').agg({'NEW_INCOME_PER_PERSON': [min, max, np.mean, np.median]})


# Feauture to show the ratio 'AMT_ANNUITY' / 'AMT_INCOME_TOTAL'
df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
# Show results for new feature
df['NEW_ANNUITY_INCOME_PERC'].head()
df['NEW_ANNUITY_INCOME_PERC'].describe()
df.groupby('TARGET').agg({'NEW_ANNUITY_INCOME_PERC': [min, max, np.mean, np.median]})


# Feature to show the ratio 'AMT_ANNUITY' / 'AMT_CREDIT'
df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
# Show results for new feature
df['NEW_PAYMENT_RATE'].head()
df['NEW_PAYMENT_RATE'].describe()
df.groupby('TARGET').agg({'NEW_PAYMENT_RATE': ['count', min, max, np.mean, np.median]})


# Feature to show the ratio 'NEW_INCOME_PER_PERSON' / 'AMT_ANNUITY'
df['NEW_INCOME_PER_PERSON_PERC_AMT_ANNUITY'] = df['NEW_INCOME_PER_PERSON'] / df['AMT_ANNUITY']
# Show results for new feature
df['NEW_INCOME_PER_PERSON_PERC_AMT_ANNUITY'].head()
df['NEW_INCOME_PER_PERSON_PERC_AMT_ANNUITY'].describe()
df.groupby('TARGET').agg({'NEW_INCOME_PER_PERSON_PERC_AMT_ANNUITY': ['count', min, max, np.mean, np.median]})


# Feature to show the ratio 'NEW_INCOME_PER_PERSON' / 'NEW_PAYMENT_RATE'
df['NEW_INCOME_PER_PERSON_PERC_PAYMENT_RATE_INCOME_PER_PERSON'] = df['NEW_INCOME_PER_PERSON'] / df['NEW_PAYMENT_RATE']
# Show results for new feature
df['NEW_INCOME_PER_PERSON_PERC_PAYMENT_RATE_INCOME_PER_PERSON'].head()
df['NEW_INCOME_PER_PERSON_PERC_PAYMENT_RATE_INCOME_PER_PERSON'].describe()
df.groupby('TARGET').agg({'NEW_INCOME_PER_PERSON_PERC_PAYMENT_RATE_INCOME_PER_PERSON': ['count', min, max, np.mean, np.median]})


# Feature that shows, if 'AMT_CREDIT' > 'AMT_GOODS_PRICE' or not.
df.loc[(df['AMT_CREDIT'] <= df['AMT_GOODS_PRICE']), 'NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'] = 0
df.loc[(df['AMT_CREDIT'] > df['AMT_GOODS_PRICE']), 'NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'] = 1
print(len(df.loc[df['NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE']==1, :])) # 230517
print(len(df.loc[df['NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE']==0, :])) # 125459
# Show results for new feature
df['NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'].head()
df['NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'].describe()
df.groupby('NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE').agg({'TARGET': ['count', 'mean']})


# Categorical age - based on target=1 plot
df['NEW_AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: feateng.get_age_label(x))
# df['NEW_AGE_RANGE'] = pd.cut(x=df['DAYS_BIRTH'] / -365, bins=[0, 27, 40, 50, 65, 99], labels=[1, 2, 3, 4, 5])
# Show results for new feature
df['NEW_AGE_RANGE'].head()
df['NEW_AGE_RANGE'].describe()
df.groupby('NEW_AGE_RANGE').agg({'TARGET': ['count', 'mean']}) # Now, we can see the differences between labels better!
# Let's visualize the results. Mean of NEW_AGE_RANGE with respect to TARGET
sns.barplot(data=df, y='TARGET', x='NEW_AGE_RANGE')
plt.show()


# Categorical working year - based on target=1 plot
df['NEW_WORKING_YEAR_RANGE'] = df['DAYS_EMPLOYED'].apply(lambda x: feateng.get_working_year_label(x))
# df['NEW_WORKING_YEAR_RANGE'] = pd.cut(x=df['DAYS_EMPLOYED'] / -365, bins=[0, 3, 5, 15, 50], labels=[1, 2, 3, 4])
# Show results for new feature
df['NEW_WORKING_YEAR_RANGE'].head()
df['NEW_WORKING_YEAR_RANGE'].describe()
df.groupby('NEW_WORKING_YEAR_RANGE').agg({'TARGET': ['count', 'mean']}) # Now, we can see the differences between labels better!
# Let's visualize the results. Mean of NEW_WORKING_YEAR_RANGE with respect to TARGET
sns.barplot(data=df, y='TARGET', x='NEW_WORKING_YEAR_RANGE')
plt.show()


# Optional: Remove 4 applications with XNA CODE_GENDER (train set)
df = df[df['CODE_GENDER'] != 'XNA']


# Optional: Remove 2 applications with Unknown NAME_FAMILY_STATUS (train set)
df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']


# Feature to show total contact information
df["NEW_TOTAL_CONTACT_INFORMATION"] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] + df['FLAG_CONT_MOBILE'] + df['FLAG_EMAIL']
df['NEW_TOTAL_CONTACT_INFORMATION'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Let's visualize the results.
df.groupby('NEW_TOTAL_CONTACT_INFORMATION').agg({'TARGET': ['count', 'mean']})
# Show results for new feature
sns.countplot(x='NEW_TOTAL_CONTACT_INFORMATION', data=df)
plt.show()
# Finding balance between classes
df.loc[(df["NEW_TOTAL_CONTACT_INFORMATION"] == 5), "NEW_TOTAL_CONTACT_INFORMATION"] = 4
df.loc[(df["NEW_TOTAL_CONTACT_INFORMATION"] == 1), "NEW_TOTAL_CONTACT_INFORMATION"] = 2
# Group by target variable
df.groupby('NEW_TOTAL_CONTACT_INFORMATION').agg({'TARGET': ['count', 'mean']})


# Feature to show last phone changing for 'NEW_YEAR_LAST_PHONE_CHANGE' variable
df["NEW_YEAR_LAST_PHONE_CHANGE"] = (df["DAYS_LAST_PHONE_CHANGE"] / -365)
# df.groupby(["NEW_YEAR_LAST_PHONE_CHANGE"]).agg({"TARGET":["count", "mean"]})

df["NEW_YEAR_LAST_PHONE_CHANGE"].max()
# Finding balance between classes
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 0) & (df["NEW_YEAR_LAST_PHONE_CHANGE"] <= 1), "NEW_YEAR_LAST_PHONE_CHANGE"] = 1
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 1) & (df["NEW_YEAR_LAST_PHONE_CHANGE"] <= 2), "NEW_YEAR_LAST_PHONE_CHANGE"] = 2
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 2) & (df["NEW_YEAR_LAST_PHONE_CHANGE"] <= 3), "NEW_YEAR_LAST_PHONE_CHANGE"] = 3
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 3) & (df["NEW_YEAR_LAST_PHONE_CHANGE"] <= 4), "NEW_YEAR_LAST_PHONE_CHANGE"] = 4
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 4) & (df["NEW_YEAR_LAST_PHONE_CHANGE"] <= 5), "NEW_YEAR_LAST_PHONE_CHANGE"] = 5
df.loc[(df["NEW_YEAR_LAST_PHONE_CHANGE"] > 5), "NEW_YEAR_LAST_PHONE_CHANGE"] = 6

df['NEW_YEAR_LAST_PHONE_CHANGE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('NEW_YEAR_LAST_PHONE_CHANGE').agg({'TARGET': ['count', 'mean']})
# Show results for new feature
sns.countplot(x='NEW_YEAR_LAST_PHONE_CHANGE', data=df)
plt.show()


# Feature to show sum of FLAG_DOCUMENTS
df["NEW_MISS_DOCUMENTS_20"] = df['FLAG_DOCUMENT_2'] + df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_4'] + \
                                df['FLAG_DOCUMENT_5'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_7'] + \
                                df['FLAG_DOCUMENT_8'] + df['FLAG_DOCUMENT_9'] + df['FLAG_DOCUMENT_10'] + \
                                df['FLAG_DOCUMENT_11'] + df['FLAG_DOCUMENT_12'] + df['FLAG_DOCUMENT_13'] + \
                                df['FLAG_DOCUMENT_14'] + df['FLAG_DOCUMENT_15'] + df['FLAG_DOCUMENT_16'] + \
                                df['FLAG_DOCUMENT_17'] + df['FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] + \
                                df['FLAG_DOCUMENT_20'] + df['FLAG_DOCUMENT_21']

df['NEW_MISS_DOCUMENTS_20'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('NEW_MISS_DOCUMENTS_20').agg({'TARGET': ['count', 'mean']})
# Show results for new feature
sns.countplot(x='NEW_MISS_DOCUMENTS_20', data=df)
plt.show()


# Feature to show finding balance between classes for 'NEW_MISS_DOCUMENTS'
df.loc[(df["NEW_MISS_DOCUMENTS_20"] == 0), "NEW_MISS_DOCUMENTS"] = 0
df.loc[(df["NEW_MISS_DOCUMENTS_20"] > 0), "NEW_MISS_DOCUMENTS"] = 1
# df['NEW_MISS_DOCUMENTS'] = df['NEW_MISS_DOCUMENTS_20'].apply(lambda x: 1 if x > 0 else 0)

df['NEW_MISS_DOCUMENTS'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('NEW_MISS_DOCUMENTS').agg({'TARGET': ['count', 'mean']})
# Show results for new feature
sns.countplot(x='NEW_MISS_DOCUMENTS', data=df)
plt.show()


# Feature to show 'NEW_AMT_REQ_CREDIT_BUREAU_YEAR'
df["NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"] + df["AMT_REQ_CREDIT_BUREAU_DAY"] + \
                                       df["AMT_REQ_CREDIT_BUREAU_WEEK"] + df["AMT_REQ_CREDIT_BUREAU_MON"] + \
                                       df["AMT_REQ_CREDIT_BUREAU_QRT"] + df["AMT_REQ_CREDIT_BUREAU_YEAR"]

df.groupby('NEW_AMT_REQ_CREDIT_BUREAU_YEAR').agg({'TARGET': ['count', 'mean']})
df.loc[(df["NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] >= 7), "NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] = 7
df['NEW_AMT_REQ_CREDIT_BUREAU_YEAR'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('NEW_AMT_REQ_CREDIT_BUREAU_YEAR').agg({'TARGET': ['count', 'mean']})
# Show results for new feature
sns.countplot(x='NEW_AMT_REQ_CREDIT_BUREAU_YEAR', hue="TARGET", data=df)
plt.show()


# Define the list for variables to drop.
drop_list = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
             'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
             'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
             'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
             'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
             'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR',
             'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
             'AMT_REQ_CREDIT_BUREAU_YEAR', 'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_AVG', 'FLOORSMIN_MEDI', 'LANDAREA_MODE',
             'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
             'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI',
             'BASEMENTAREA_MEDI', 'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE', 'NONLIVINGAPARTMENTS_MODE',
             'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI', 'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI',
             'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE', 'COMMONAREA_MEDI',
             'ELEVATORS_MODE', 'WALLSMATERIAL_MODE', "APARTMENTS_MODE", 'APARTMENTS_MEDI', 'APARTMENTS_AVG', 'ENTRANCES_AVG', 'ENTRANCES_MEDI',
             'ENTRANCES_MODE', 'LIVINGAREA_MODE', 'LIVINGAREA_AVG', 'LIVINGAREA_MEDI', 'HOUSETYPE_MODE', 'FLOORSMAX_AVG', 'FLOORSMAX_MEDI',
             'FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'TOTALAREA_MODE', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
             'LIVE_REGION_NOT_WORK_REGION']


# MISSING VALUES ANALYSIS

# Examine missing values for dataframe
feateng.missing_values_table(df)

# Visualize missing variables
plt.figure(figsize=(32, 32))
sns.heatmap(df.isnull(), cbar=False, cmap='magma')
plt.show()

# Missing values overall view
msno.bar(df)
plt.show()

# Now, we can see the relationship between missing values
msno.matrix(df)
plt.show()

# Nullity correlation visualization
msno.heatmap(df)
plt.show()

# Missing values will not be handled/treated here, because we are going to use XGBoost an LightGBM.
# These are robust to missing values and outliers.


# OUTLIER ANALYSIS

# Check for outliers (Quantiles can be changed and the new stituation can be observed)
# (There are some variables, that seem to be numerical, but in fact categorical. These must be excluded!)
feateng.has_outliers(df, num_cols, plot=False)

# Catch more_cat_cols and analyse more_cat_cols and num_cols more intensively.
more_cat_cols = [col for col in df.columns if df[col].nunique() < 30 and col not in 'TARGET' and not col in cat_cols]
print('Number of More Categorical Variables : ', len(more_cat_cols))
print(more_cat_cols)
print('Number of Numerical Variables : ', len(num_cols))
print(num_cols)

# Now, it is time to exclude variables, that seem to be numerical, but in fact categorical
less_num_cols = [col for col in num_cols if not col in more_cat_cols]
print('Number of Less Numerical Variables : ', len(less_num_cols))
print(less_num_cols)

# Check for outliers, again
feateng.has_outliers(df, less_num_cols, plot=False)

# Outliers will not be handled/treated here, because we are going to use XGBoost an LightGBM.
# These are robust to missing values and outliers.


# LABEL AND ONE HOT ENCODING

df, cat_cols_after_ohe = feateng.one_hot_encoder(df, cat_cols)
print('Number of new features: {}'.format(len(cat_cols_after_ohe)))


# FEATURE SCALING

# Because we do not use distance based classification algorithm, we do not implement scaling.
# By modeling we will use XGBoost and/or LightGBM


# CONTROL THE DATASET

# Last look at the prepared dataset.
df.head()
df.describe()
df.info()
# Check for data types
df.dtypes


# SAVE DATASET FOR MODELING

# import snappy
import fastparquet

# Saving the Dataset as a parquet file.
df.to_parquet("data_preprocessed.parquet", compression='gzip')



