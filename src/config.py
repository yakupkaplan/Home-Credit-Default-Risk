"""
Config File for Home Credit Default Risk Prediction
"""

# Define paths for the data sources
PATH_APPLICATION_TRAIN = "data/application_train.csv"
PATH_APPLICATION_TEST = "data/application_test.csv"
PATH_BUREAU = "data/bureau.csv"
PATH_BUREAU_BALANCE = "data/bureau_balance.csv"
PATH_CREDIT_CARD_BALANCE = "data/credit_card_balance.csv"
PATH_INSTALLMENTS_PAYMENTS = "data/installments_payments.csv"
PATH_POS_CASH_BALANCE = "data/POS_CASH_balance.csv"
PATH_PREVIOUS_APPLICATION = "data/previous_application.csv"

# Paths for prepared datasets
FINAL_TRAIN_DF = "data/final_train_df.pkl"
FINAL_TEST_DF = "data/final_test_df.pkl"

# Prediction file
PREDICTIONS = "results/predictions_submission.csv"
