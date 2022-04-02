"""
PREPROCESSING FOR HOME CREDIT DEFAULT PREDICTION
"""

import gc
import pandas as pd
import numpy as np

from src.helper_functions import one_hot_encoder, rare_encoder
from src.config import *


def feature_eng_application_train(df):
    """
    Feature Engineering steps for application_train and application_test.
    Returns dataframe with FE steps implemented.

    :param df: dataframe
        dataframe(application_train) for FE-steps

    :return: dataframe

    """
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set) and Remove 2 applications with Unknown NAME_FAMILY_STATUS (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Some simple new features (percentages)
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_LOAN_VALUE_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_INCOME_PER_PERSON_PERC_AMT_ANNUITY'] = df['NEW_INCOME_PER_PERSON'] / df['AMT_ANNUITY']
    df['NEW_INCOME_PER_PERSON_PERC_PAYMENT_RATE_INCOME_PER_PERSON'] = df['NEW_INCOME_PER_PERSON'] / df[
        'NEW_PAYMENT_RATE']
    df.loc[(df['AMT_CREDIT'] <= df['AMT_GOODS_PRICE']), 'NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'] = 0
    df.loc[(df['AMT_CREDIT'] > df['AMT_GOODS_PRICE']), 'NEW_FLAG_CREDIT_MORE_THAN_GOODSPRICE'] = 1
    df['NEW_AGE_RANGE'] = pd.cut(x=df['DAYS_BIRTH'] / -365, bins=[0, 27, 40, 50, 65, 99], labels=[1, 2, 3, 4, 5])
    df['NEW_WORKING_YEAR_RANGE'] = pd.cut(x=df['DAYS_EMPLOYED'] / -365, bins=[0, 3, 5, 15, 50], labels=[1, 2, 3, 4])
    df["NEW_TOTAL_CONTACT_INFORMATION"] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] + df[
        'FLAG_CONT_MOBILE'] + df['FLAG_EMAIL']
    df['NEW_YEAR_LAST_PHONE_CHANGE'] = pd.cut(x=df['DAYS_LAST_PHONE_CHANGE'] / -365,
                                              bins=[-0.1, 0, 1, 2, 3, 4, 5, 11.75], labels=[0, 1, 2, 3, 4, 5, 6])
    df["NEW_MISS_DOCUMENTS_20"] = df['FLAG_DOCUMENT_2'] + df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_4'] + \
                                  df['FLAG_DOCUMENT_5'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_7'] + \
                                  df['FLAG_DOCUMENT_8'] + df['FLAG_DOCUMENT_9'] + df['FLAG_DOCUMENT_10'] + \
                                  df['FLAG_DOCUMENT_11'] + df['FLAG_DOCUMENT_12'] + df['FLAG_DOCUMENT_13'] + \
                                  df['FLAG_DOCUMENT_14'] + df['FLAG_DOCUMENT_15'] + df['FLAG_DOCUMENT_16'] + \
                                  df['FLAG_DOCUMENT_17'] + df['FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] + \
                                  df['FLAG_DOCUMENT_20'] + df['FLAG_DOCUMENT_21']
    df['NEW_FLAG_MISS_DOCUMENTS'] = df['NEW_MISS_DOCUMENTS_20'].apply(lambda x: 1 if x > 0 else 0)
    df["NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"] + df["AMT_REQ_CREDIT_BUREAU_DAY"] + \
                                           df["AMT_REQ_CREDIT_BUREAU_WEEK"] + df["AMT_REQ_CREDIT_BUREAU_MON"] + \
                                           df["AMT_REQ_CREDIT_BUREAU_QRT"] + df["AMT_REQ_CREDIT_BUREAU_YEAR"]
    df.loc[(df["NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] >= 7), "NEW_AMT_REQ_CREDIT_BUREAU_YEAR"] = 7

    # Define the list for variables to drop.
    drop_list = ['FLAG_DOCUMENT_9', 'LANDAREA_MODE', 'FLAG_WORK_PHONE', 'FLAG_DOCUMENT_8', 'FLOORSMIN_MEDI',
                 'ELEVATORS_MODE', 'COMMONAREA_MODE', 'NONLIVINGAPARTMENTS_AVG',
                 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_19', 'AMT_REQ_CREDIT_BUREAU_MON', 'NONLIVINGAPARTMENTS_MEDI',
                 'REG_REGION_NOT_LIVE_REGION', 'FLAG_DOCUMENT_16', 'ENTRANCES_MODE', 'CNT_FAM_MEMBERS',
                 'ENTRANCES_MEDI', 'YEARS_BUILD_MEDI', 'YEARS_BEGINEXPLUATATION_AVG', 'FLAG_DOCUMENT_7',
                 'FLOORSMAX_AVG', 'FLAG_DOCUMENT_18', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'LIVINGAPARTMENTS_MODE',
                 'FLAG_DOCUMENT_2', 'BASEMENTAREA_MODE', 'BASEMENTAREA_MEDI', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_10',
                 'ELEVATORS_AVG', 'YEARS_BUILD_MODE', 'COMMONAREA_AVG', 'AMT_REQ_CREDIT_BUREAU_DAY', 'LIVINGAREA_MODE',
                 'FLAG_DOCUMENT_3', 'LANDAREA_MEDI', 'DAYS_LAST_PHONE_CHANGE',
                 'REG_REGION_NOT_WORK_REGION', 'COMMONAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
                 'REGION_RATING_CLIENT_W_CITY', 'FLAG_DOCUMENT_5', 'APARTMENTS_MEDI', 'LIVINGAPARTMENTS_AVG',
                 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 'NONLIVINGAREA_MEDI', 'FLAG_DOCUMENT_21',
                 'YEARS_BEGINEXPLUATATION_MODE', 'APARTMENTS_AVG', 'ENTRANCES_AVG', 'FLAG_PHONE',
                 'LIVE_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_6', 'BASEMENTAREA_AVG', 'FLAG_DOCUMENT_14',
                 'FLAG_EMP_PHONE', 'NONLIVINGAPARTMENTS_MODE', 'FLOORSMIN_AVG', 'FLAG_MOBIL', 'LIVINGAREA_AVG',
                 'FLAG_CONT_MOBILE', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'YEARS_BUILD_AVG', 'NONLIVINGAREA_AVG',
                 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_4', 'AMT_REQ_CREDIT_BUREAU_QRT', 'LIVINGAPARTMENTS_MEDI',
                 'FLAG_DOCUMENT_11', 'NONLIVINGAREA_MODE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_20', 'FLOORSMIN_MODE', 'FLAG_EMAIL',
                 'OBS_60_CNT_SOCIAL_CIRCLE', 'ELEVATORS_MEDI', 'LANDAREA_AVG', 'APARTMENTS_MODE', 'FLAG_DOCUMENT_17',
                 'LIVINGAREA_MEDI', 'TOTALAREA_MODE',
                 'EMERGENCYSTATE_MODE', 'WALLSMATERIAL_MODE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE']
    df.drop(drop_list, axis=1, inplace=True)

    return df


def application_train_test(num_rows=None, nan_as_category=False):
    """
    Loads, merges datasets. Afterwards, feature_eng_application_train and one_hot_encoder functions are called.
    Returns dataframe with one-hot encoded and feature engineering implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe

    """
    # Read data and merge
    df = pd.read_csv(PATH_APPLICATION_TRAIN, nrows=num_rows)
    test_df = pd.read_csv(PATH_APPLICATION_TEST, nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # Apply feature engineering for application_train
    df = feature_eng_application_train(df)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    del test_df
    gc.collect()

    return df


def feature_eng_bureau_and_balance(df):
    """
    Feature Engineering steps for bureau_and_balance.
    Returns dataframe with FE steps implemented.

    :param df: dataframe
        dataframe(bureau_and_balance) for FE-steps

    :return:dataframe

    """
    df.fillna(0, inplace=True)

    grp = df[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(
        index=str, columns={'DAYS_CREDIT': 'NEW_BUREAU_LOAN_COUNT'})
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(
        index=str, columns={'CREDIT_TYPE': 'NEW_BUREAU_LOAN_TYPES'})
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')

    df['CREDIT_ACTIVE_BINARY'] = df['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Active' else 0)
    grp = df.groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ACTIVE_BINARY': 'NEW_ACTIVE_LOANS_PERCENTAGE'})
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')
    del df['CREDIT_ACTIVE_BINARY']
    gc.collect()

    df['CREDIT_ENDDATE_BINARY'] = df['DAYS_CREDIT_ENDDATE'].apply(lambda x: 0 if x <= 0 else 1)
    grp = df.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ENDDATE_BINARY': 'NEW_CREDIT_ENDDATE_PERCENTAGE'})
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')
    del df['CREDIT_ENDDATE_BINARY']
    gc.collect()

    grp1 = df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().\
        reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = df[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
    df = df.merge(grp1, on=['SK_ID_CURR'], how='left')
    df = df.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()

    df['NEW_DEBT_CREDIT_RATIO'] = df['TOTAL_CUSTOMER_DEBT'] / df['TOTAL_CUSTOMER_CREDIT']
    del df['TOTAL_CUSTOMER_DEBT'], df['TOTAL_CUSTOMER_CREDIT']
    gc.collect()

    return df


def aggregations_bureau_and_balance(bureau, bb, bureau_cat, bb_cat):
    """
    Aggregation operations for bureau_and_balance
    Returns dataframe after completing specific aggregations for numerical and categorical variables for bureau
    and bureau_balance tables and finally joins these two tables.

    :param bureau: dataframe
        dataframe(bureau) for applying aggregations

    :param bb: dataframe
        dataframe(bureau_balance) for applying aggregations

    :param bureau_cat: list
        list that holds categorical variables for bureau table

    :param bb_cat: list
        list that holds categorical variables for bureau_balance table

    :return: dataframe

    """
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'BB_MONTHS_BALANCE_MIN': ['min'],
        'BB_MONTHS_BALANCE_MAX': ['max'],
        'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'NEW_BUREAU_LOAN_COUNT': ['mean'],
        'NEW_BUREAU_LOAN_TYPES': ['mean'],
        'NEW_ACTIVE_LOANS_PERCENTAGE': ['max', 'mean'],
        'NEW_CREDIT_ENDDATE_PERCENTAGE': ['max', 'mean'],
        'NEW_DEBT_CREDIT_RATIO': ['max', 'mean']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations["BB_" + cat + "_MEAN"] = ['mean']
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    return bureau_agg


def bureau_and_balance(num_rows=None, nan_as_category=True):
    """
    Loads, merges datasets. Afterwards, feature_eng_application_train, rare_encoding, one_hot_encoder and aggregations
    functions are called.
    Returns dataframe with one-hot encoded, rare-encoded, feature engineering and aggregations implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe

    """
    # Load the datasets
    bureau = pd.read_csv(PATH_BUREAU, nrows=num_rows)
    bb = pd.read_csv(PATH_BUREAU_BALANCE, nrows=num_rows)

    # Apply feature engineering steps for bureau_and_balance
    bureau = feature_eng_bureau_and_balance(bureau)

    # Implement rare encoding
    bureau = rare_encoder(bureau, 0.01)

    # Apply one hot encoding
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=nan_as_category)

    # Apply aggregation operations to the dataset
    bureau_agg = aggregations_bureau_and_balance(bureau, bb, bureau_cat, bb_cat)

    return bureau_agg


def feature_eng_previous_applications(df):
    """
    Feature Engineering steps for previous_applications.
    Returns dataframe with FE steps implemented.

    :param df: dataframe
        dataframe(previous_applications) for FE-steps

    :return:dataframe

    """
    accompanied = ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
    df["NAME_TYPE_SUITE"] = df["NAME_TYPE_SUITE"].replace(accompanied, 'Accompanied')

    # Otherization
    name_others = ['Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure',
                   'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness',
                   'Additional Service', 'Education', 'Weapon', 'Insurance', 'House Construction', 'Animals']
    df["NAME_GOODS_CATEGORY"] = df["NAME_GOODS_CATEGORY"].replace(name_others, 'others')

    channel_others = ['AP+ (Cash loan)', 'Channel of corporate sales', 'Car dealer']
    df["CHANNEL_TYPE"] = df["CHANNEL_TYPE"].replace(channel_others, 'Other_Channel')

    seller_others = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism']
    df["NAME_SELLER_INDUSTRY"] = df["NAME_SELLER_INDUSTRY"].replace(seller_others, 'Others')

    loan_others = ['Refusal to name the goal', 'Money for a third person', 'Buying a garage',
                   'Gasification / water supply',
                   'Hobby', 'Business development', 'Buying a holiday home / land', 'Furniture', 'Car repairs',
                   'Buying a home', 'Wedding / gift / holiday']
    df["NAME_CASH_LOAN_PURPOSE"] = df["NAME_CASH_LOAN_PURPOSE"].replace(loan_others, 'Other_Loan')

    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    df['NEW_APP_CREDIT_RATE'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
    df["NEW_APP_CREDIT_RATE_RATIO"] = df["NEW_APP_CREDIT_RATE"].apply(lambda x: 1 if (x <= 1) else 0)
    df['NEW_AMT_PAYMENT_RATE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_APP_GOODS_RATE'] = df['AMT_APPLICATION'] / df['AMT_GOODS_PRICE']
    df['NEW_CREDIT_GOODS_RATE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_RETURN_DAY'] = df['DAYS_DECISION'] + df['CNT_PAYMENT'] * 30
    df['NEW_DAYS_TERMINATION_DIFF'] = df['DAYS_TERMINATION'] - df['NEW_RETURN_DAY']
    df['NEW_DAYS_DUE_DIFF'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_FIRST_DUE']
    df["NEW_CNT_PAYMENT"] = pd.cut(x=df['CNT_PAYMENT'], bins=[0, 12, 60, 120], labels=["Short", "Middle", "Long"])
    df["NEW_END_DIFF"] = df["DAYS_TERMINATION"] - df["DAYS_LAST_DUE"]

    weekend = ["SATURDAY", "SUNDAY"]
    df["WEEKDAY_APPR_PROCESS_START"] = df["WEEKDAY_APPR_PROCESS_START"].apply(lambda x: "WEEKEND" if (x in weekend) else "WEEKDAY")

    df['NFLAG_LAST_APPL_IN_DAY'] = df['NFLAG_LAST_APPL_IN_DAY'].astype("O")
    df['FLAG_LAST_APPL_PER_CONTRACT'] = df['FLAG_LAST_APPL_PER_CONTRACT'].astype("O")
    df["NEW_CNT_PAYMENT"] = df['NEW_CNT_PAYMENT'].astype("O")
    df['NEW_APP_CREDIT_RATE_RATIO'] = df['NEW_APP_CREDIT_RATE_RATIO'].astype('O')
    new_coding = {"0": "Yes", "1": "No"}
    df['NEW_APP_CREDIT_RATE_RATIO'] = df['NEW_APP_CREDIT_RATE_RATIO'].replace(new_coding)

    return df


def aggregations_previous_applications(df, cat_cols):
    """
    Aggregation operations for previous_applications
    Returns dataframe after completing specific aggregations for numerical and categorical variables for previous_applications.

    :param df: dataframe
        dataframe(bureau) for applying aggregations

    :param cat_cols: list
        list that holds categorical variables for bureau table

    :return: dataframe

    """
    # Aggregation for numeric features
    num_aggregations = {
        'SK_ID_PREV': 'count',
        'AMT_ANNUITY': ['min', 'max', 'median', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'median'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'median'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'median'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'DAYS_DECISION': ['min', 'max', 'mean', 'median'],
        'NEW_APP_CREDIT_RATE': ['min', 'max', 'mean', 'var'],
        'NEW_AMT_PAYMENT_RATE': ['min', 'max', 'mean'],
        'NEW_APP_GOODS_RATE': ['min', 'max', 'mean'],
        'NEW_CREDIT_GOODS_RATE': ['min', 'max', 'mean'],
        'NEW_RETURN_DAY': ['min', 'max', 'mean', 'var'],
        'NEW_DAYS_TERMINATION_DIFF': ['min', 'max', 'mean'],
        'NEW_END_DIFF': ['min', 'max', 'mean'],
        'NEW_APP_CREDIT_RATE_RATIO': ['min', 'max', 'mean'],
        'NEW_DAYS_DUE_DIFF': ['min', 'max', 'mean']
    }

    # Aggregation for categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Approved Applications - Aggregation for numeric features
    approved = df[df['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Refused Applications - Aggregation for numeric features
    refused = df[df['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, df
    gc.collect()

    return prev_agg


def previous_applications(num_rows=None, nan_as_category=True):
    """
    Loads previous_applications dataset. Afterwards, feature_eng_application_train, one_hot_encoder and aggregations
    functions are called.
    Returns dataframe with one-hot encoded, feature engineering and aggregations implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe

    """
    df_prev = pd.read_csv(PATH_PREVIOUS_APPLICATION, nrows=num_rows)

    # Implement feature engineering operations for df_prev
    df_prev = feature_eng_previous_applications(df_prev)

    # Apply one hot encoding
    df_prev, cat_cols = one_hot_encoder(df_prev, nan_as_category=nan_as_category)

    # Apply aggregation operations to the dataset
    prev_agg = aggregations_previous_applications(df_prev, cat_cols)

    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    """
    Loads pos_cash dataset. Afterwards, one_hot_encoder and aggregations steps are implemented.
    Returns dataframe with one-hot encoded and aggregations implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe

    """
    pos = pd.read_csv(PATH_POS_CASH_BALANCE, nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(num_rows=None, nan_as_category=True):
    """
    Loads installments_payments dataset. Afterwards, one_hot_encoder, feature engineering and aggregations steps are implemented.
    Returns dataframe with one-hot encoded, feature engineering and aggregations implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe
    """
    ins = pd.read_csv(PATH_INSTALLMENTS_PAYMENTS, nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=nan_as_category)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(num_rows=None, nan_as_category=True):
    """
    Loads credit_card_balance dataset. Afterwards, one_hot_encoder, feature engineering and aggregations steps are implemented.
    Returns dataframe with one-hot encoded, feature engineering and aggregations implemented columns.

    :param num_rows: int
        int that shows number of rows to be loaded for the dataset

    :param nan_as_category: bool
        boolean that shows, if nan values will be created as separate columns or not.

    :return: dataframe
    """
    cc = pd.read_csv(PATH_CREDIT_CARD_BALANCE, nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc
    gc.collect()

    return cc_agg
