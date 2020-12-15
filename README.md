# Home-Credit-Default-Risk

In this project we try to predict home credit default risk for clients. We try to predict, if the client will have payment difficulties or not.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/overview

Reference notebook: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

__Project team members:__

- Ali Baltaci: https://github.com/alibaltaci
- Beren Sak: https://github.com/berensak
- Yakup Kaplan: https://github.com/yakupkaplan

__Mentor:__

- Muhammet Cakmak: https://github.com/muhammet-cakmak


__Target definition:__ 

1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample,              
0 - all other cases


__Steps to follow for EDA:__

- General View
- Categorical Variables Analysis
- More Categorical Variables Analysis (Variables, that seem to be numerical, but in fact they have low range of labels and can be thought as categorical variables.)
- Numerical Variables Analysis
- Target Analysis
- Feature by Feature EDA

__Steps to follow for data preprocessing and feature engineering:__

- New Features Creation and Analysis of New Features
- Missing Values Analysis, but not treatment
- Outlier Analysis, but not treatment
- Label and One Hot Encoding
- Standardization / Feature Scaling
- Control the Dataset
- Save Dataset for Modeling

## Description

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Data Description

__application_{train|test}.csv__
- This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
- Static data for all applications. One row represents one loan in our data sample.

__bureau.csv__

- All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
- For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

__bureau_balance.csv__

- Monthly balances of previous credits in Credit Bureau.
- This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

__POS_CASH_balance.csv__

- Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

__credit_card_balance.csv__

- Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

__previous_application.csv__

- All previous applications for Home Credit loans of clients who have loans in our sample.
- There is one row for each previous application related to loans in our data sample.

__installments_payments.csv__

- Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
- There is a) one row for every payment that was made plus b) one row each for missed payment.
- One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

![Home Credit Tables](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
