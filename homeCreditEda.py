# HOME CREDIT DEFAULT RISK PROJECT

# EXPLORATORY DATA ANALYSIS

'''
In this project we try to predict home credit default risk for clients.
In this script we focus on exploratory data analysis.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/overview

Steps to follow for EDA:
    - General View
    - Categorical Variables Analysis
    - More Categorical Variables Analysis (Variables, that seem to be numerical, but in fact they have low range of labels and can be thought as categorical variables.)
    - Numerical Variables Analysis
    - Target Analysis
    - Feature by Feature EDA

'''


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

# Import functions from EdaUtills.py.
import EdaUtills as eda


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

df = application_train.append(application_test)


## GENERAL VIEW

# First look into the dataset
df.head()
print(df.shape) # (307511, 122)
df.info()
print(df.columns)
print(df.index)


# Check for missing values
df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)


# Number of TARGET labels
df['TARGET'].value_counts()
print('Ratio of risky customers = {}' .format(df['TARGET'].value_counts()[1] / df['TARGET'].value_counts()[0]))
# Ratio of risky customers = 0.08781828601345662

# See how many 0 and 1 values in the dataset and if there is imbalance
sns.countplot(x='TARGET', data=df)
plt.show()


## CATEGORICAL VARIABLES ANALYSIS

# Let's catch categorical variables
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
print('Number of Categorical Variables : ', len(cat_cols))

# Show summary for categorical variables
eda.cat_summary(df, cat_cols, 'TARGET')

# numbers of unique classes for each cat_cols
df[cat_cols].nunique()

# Number of children
eda.cat_summary(df, ['CNT_CHILDREN'], 'TARGET')


## MORE CAT COLS ANALYSIS

# There are some variables, that seem to be numerical, but in fact they have low range of labels and can be thought as categorical variables.

# Let's catch more_cat_cols
more_cat_cols = [col for col in application_train.columns if application_train[col].nunique() < 30 and col not in 'TARGET' and not col in cat_cols]
print('Number of More Categorical Variables: ', len(more_cat_cols))
print(more_cat_cols)

# Show summary for more categorical variables
eda.cat_summary(df, more_cat_cols, 'TARGET', number_of_classes=30)


## NUMERICAL VARIABLES ANALYSIS

# Let's catch numerical variables
num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ['SK_ID_CURR', 'TARGET']]
print('Number of Numerical Variables : ', len(num_cols))

# Show histogtams for numerical variables
eda.hist_for_nums(df, num_cols)

# Group by target variable
for col in num_cols:
    print(df.groupby('TARGET').agg({col: ['min', 'max', 'mean', 'median']}))


# Distribution plots by adding hue = 'TARGET'
for col in num_cols:
    sns.displot(df, x=col, hue="TARGET", bins=20, multiple="dodge") # 'stack'
    plt.show()

df[num_cols].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

# Boxplots
for col in num_cols:
    sns.boxplot(data=df, x='TARGET', y=col, )
    plt.show()

# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(50, 50))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, fmt='.1g', cbar_kws={"shrink": .5})
plt.show()

# Heatmap for correlation for the ones with high correlation rate. (|r|>=0.3)
dfCorr = df.corr()
filteredDf = dfCorr[((dfCorr >= .3) | (dfCorr <= -.3)) & (dfCorr !=1.000)]
plt.figure(figsize=(50, 50))
sns.heatmap(filteredDf, annot=True, fmt='.1g', cmap="Reds")
plt.show()


## TARGET ANALYSIS

# Number of TARGET labels
df['TARGET'].value_counts()

# See how many 0 and 1 values in the dataset and if there is imbalance
sns.countplot(x='TARGET', data=df)
plt.show()

# See the correlations with respect to 'TARGET'
low_correlations, high_correlations = eda.find_correlation(df, num_cols, corr_limit=0.20)
print(high_correlations) # There are no high correlation with 'TARGET'


## FEATURE BY FEATURE EDA

'''
# Steps to follow for each variable:
Follow the steps for important variables.
    - description
    - cat_summary, visualization,
    - histogram, visualization, visualization with respect to each other
    - new features, examine new feature
    - 
'''

# 'TARGET': Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)
df.loc[df['TARGET']==1, :][:10]
df.loc[df['TARGET']==0, :][:10]

# 'NAME_CONTRACT_TYPE': Identification if loan is cash or revolving.
# Cash loans(90%), revolving loans(10%)
# Installment credit is an extension of credit by which fixed, scheduled payments are made until
# the loan is paid in full. Revolving credit is credit that is renewed as the debt is paid, allowing
# the borrower access to a line of credit when needed
df['NAME_CONTRACT_TYPE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('NAME_CONTRACT_TYPE').agg({'TARGET': ['count', 'mean']})

# 'CODE_GENDER': Gender of the client
df['CODE_GENDER'].value_counts() # 4 XNA can be removed.
df.groupby('CODE_GENDER').agg({'TARGET': ['count', 'mean']}) # There is a difference between classes.

# 'FLAG_OWN_CAR': Flag if the client owns a car
df['FLAG_OWN_CAR'].value_counts() # Mostly, no car.
df.groupby('FLAG_OWN_CAR').agg({'TARGET': ['count', 'mean']}) # No difference.

# 'FLAG_OWN_REALTY': Flag if client owns a house or flat
df['FLAG_OWN_REALTY'].value_counts() # Most of the clients have a house or flat.
df.groupby('FLAG_OWN_REALTY').agg({'TARGET': ['count', 'mean']}) # No difference.

# 'CNT_CHILDREN': Number of children the client has --> Watch out for outliers, anomalies, strange relations after >=5
df['CNT_CHILDREN'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('CNT_CHILDREN').agg({'TARGET': ['count', 'mean']}) # A new class like 5+ can be created for rares.

# AMT_INCOME_TOTAL : Income of the client (monthly)--> Outliers!!!
df['AMT_INCOME_TOTAL'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T # Outliers can be removed! --> df = df[df['AMT_INCOME_TOTAL'] < 20000000]
df.groupby('TARGET').agg({'AMT_INCOME_TOTAL': ['count', 'min', 'max', 'mean', 'median']})
df.sort_values('AMT_INCOME_TOTAL', ascending=True).head(15)
df.sort_values('AMT_INCOME_TOTAL', ascending=False).head(15)
# Mean of Incomes with respect to REGION_RATING_CLIENT
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='REGION_RATING_CLIENT', )
plt.show()

# AMT_CREDIT: Credit amount of the loan
df['AMT_CREDIT'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('TARGET').agg({'AMT_CREDIT': ['count', 'min', 'max', 'mean', 'median']})
# Mean of Incomes with respect to REGION_RATING_CLIENT
sns.barplot(data=df, y='AMT_CREDIT', x='REGION_RATING_CLIENT', )
plt.show()

# 'AMT_INCOME_TOTAL' and 'AMT_CREDIT' together
df[['AMT_CREDIT', 'AMT_GOODS_PRICE']].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df[['AMT_CREDIT', 'AMT_GOODS_PRICE']].head(10)
df.groupby('TARGET').agg({'AMT_CREDIT': ['count', 'min', 'max', 'mean', 'median'], 'AMT_GOODS_PRICE': ['count', 'min', 'max', 'mean', 'median']})
# Number of customers, whose credit amount equals to the price of the goods for which the loan is given
len(application_train.loc[application_train['AMT_CREDIT'] == application_train['AMT_GOODS_PRICE'], :]) # 108210
# Number of customers, whose credit amount NOT equals to the price of the goods for which the loan is given
len(application_train.loc[application_train['AMT_CREDIT'] != application_train['AMT_GOODS_PRICE'], :]) # 199301
# Number of customers, whose credit amount more than the price of the goods for which the loan is given
len(application_train.loc[application_train['AMT_CREDIT'] > application_train['AMT_GOODS_PRICE'], :]) # 198763
# Number of customers, whose credit amount less than the price of the goods for which the loan is given
len(application_train.loc[application_train['AMT_CREDIT'] < application_train['AMT_GOODS_PRICE'], :]) # 260
# Number of customers, whose credit amount less than the price of the goods for which the loan is given and 'TARGET' == 0
len(application_train.loc[(application_train['AMT_CREDIT'] < application_train['AMT_GOODS_PRICE']) & (application_train['TARGET']==0), :]) # 247
# Number of customers, whose credit amount less than the price of the goods for which the loan is given and 'TARGET' == 1
len(application_train.loc[(application_train['AMT_CREDIT'] < application_train['AMT_GOODS_PRICE']) & (application_train['TARGET']==1), :])  # 13

# 'AMT_ANNUITY': Loan annuity
df['AMT_ANNUITY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'AMT_ANNUITY': ['count', 'min', 'max', 'mean', 'median']})
# Mean of Incomes with respect to REGION_RATING_CLIENT
sns.barplot(data=df, y='AMT_ANNUITY', x='REGION_RATING_CLIENT', )
plt.show()

# 'AMT_GOODS_PRICE': For consumer loans it is the price of the goods for which the loan is given
df['AMT_GOODS_PRICE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'AMT_GOODS_PRICE': ['count', 'min', 'max', 'mean', 'median']})
# Mean of AMT_GOODS_PRICE with respect to NAME_INCOME_TYPE
sns.barplot(data=df, y='AMT_GOODS_PRICE', x='NAME_INCOME_TYPE', )
plt.show()

# 'NAME_TYPE_SUITE': Who was accompanying client when he was applying for the loan
# 'Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_A', nan, 'Other_B', 'Group of people'
df['NAME_TYPE_SUITE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('NAME_TYPE_SUITE').agg({'TARGET': ['count', 'mean']})
# Mean of TARGET with respect to NAME_TYPE_SUITE
sns.barplot(data=df, y='TARGET', x='NAME_TYPE_SUITE', )
plt.show()

# 'NAME_INCOME_TYPE': Clients income type
# 'Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed', 'Student', 'Businessman', 'Maternity leave'
df['NAME_INCOME_TYPE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('NAME_INCOME_TYPE').agg({'TARGET': ['count', 'mean']}) # No default by Businessman and Student!
# Mean of TARGET with respect to NAME_INCOME_TYPE
sns.barplot(data=df, y='TARGET', x='NAME_INCOME_TYPE', )
plt.show()
# Mean of AMT_INCOME_TOTAL with respect to NAME_TYPE_SUITE
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='NAME_INCOME_TYPE', hue="TARGET")
plt.show()

# 'NAME_EDUCATION_TYPE': Level of highest education the client achieved
# 'Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'
df['NAME_EDUCATION_TYPE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('NAME_EDUCATION_TYPE').agg({'TARGET': ['count', 'mean']}) # Higher education level, lower default risk!
# Mean of TARGET with respect to NAME_EDUCATION_TYPE
sns.barplot(data=df, y='TARGET', x='NAME_EDUCATION_TYPE')
plt.show()
# Mean of AMT_INCOME_TOTAL with respect to NAME_EDUCATION_TYPE
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='NAME_EDUCATION_TYPE', hue="TARGET")
plt.show()
# Mean of AMT_CREDIT with respect to NAME_EDUCATION_TYPE
sns.barplot(data=df, y='AMT_CREDIT', x='NAME_EDUCATION_TYPE', hue="TARGET")
plt.show()

# 'NAME_FAMILY_STATUS': Family status of the client ('Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated', 'Unknown')
df['NAME_FAMILY_STATUS'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('NAME_FAMILY_STATUS').agg({'TARGET': ['count', 'mean']}) # Differences between classes! --> Unknown can be removed!
# Mean of TARGET with respect to NAME_FAMILY_STATUS
sns.barplot(data=df, y='TARGET', x='NAME_FAMILY_STATUS')
plt.show()
# Mean of AMT_INCOME_TOTAL with respect to NAME_FAMILY_STATUS
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='NAME_FAMILY_STATUS', hue="TARGET")
plt.show()

# 'NAME_HOUSING_TYPE': What is the housing situation of the client
# ('House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment')
df['NAME_HOUSING_TYPE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('NAME_HOUSING_TYPE').agg({'TARGET': ['count', 'mean']}) # Differences between classes! --> 'Rented apartment' and 'With parents' default risk higher!
# Mean of TARGET with respect to NAME_FAMILY_STATUS
sns.barplot(data=df, y='TARGET', x='NAME_HOUSING_TYPE')
plt.show()
# Mean of AMT_INCOME_TOTAL with respect to NAME_FAMILY_STATUS
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='NAME_HOUSING_TYPE', hue="TARGET")
plt.show()

# 'REGION_POPULATION_RELATIVE': Normalized population of region where client lives (higher number means the client lives in more populated region)
df['REGION_POPULATION_RELATIVE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'REGION_POPULATION_RELATIVE': ['count', 'min', 'max', 'mean', 'median']}) # No difference!

# 'DAYS_BIRTH': Client's age in days at the time of application --> Categorization/Discretization can be implemented!
df['DAYS_BIRTH'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'DAYS_BIRTH': ['count', 'min', 'max', 'mean', 'median']})

# 'DAYS_EMPLOYED': How many days before the application the person started current employment
df['DAYS_EMPLOYED'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'DAYS_EMPLOYED': ['count', 'min', 'max', 'mean', 'median']}) # The ones with higher days employed have lower default risk!

print('Number of days > 0: ', len(application_train.loc[application_train['DAYS_EMPLOYED'] >0, :])) # Pensioner + Unemployed - Pensioner, but still working
print('Number of days < 0: ', len(application_train.loc[application_train['DAYS_EMPLOYED'] <0, :])) # Still working + Unemployed - Pensioner, but still working
print('Number of days = 0: ', len(application_train.loc[application_train['DAYS_EMPLOYED'] == 0, :])) # customers who started today

print('Number of pensioners: ', len(application_train.loc[application_train['NAME_INCOME_TYPE'] == 'Pensioner', :]) ) # Pensioners
print('Number of customers who are still working: ', len(application_train.loc[application_train['NAME_INCOME_TYPE'] != 'Pensioner', :]) ) # Still working + Unemployed
print('Number of customers who are unemployed: ', len(application_train.loc[application_train['NAME_INCOME_TYPE'] == 'Unemployed', :]) ) # Unemployed
# Today is the first day at the new job!
application_train.loc[application_train['DAYS_EMPLOYED'] == 0, :]

# 'DAYS_REGISTRATION': How many days before the application did client change his registration
df['DAYS_REGISTRATION'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'DAYS_REGISTRATION': ['count', 'min', 'max', 'mean', 'median']})

# 'DAYS_ID_PUBLISH': How many days before the application did client change the identity document with which he applied for the loan
df['DAYS_ID_PUBLISH'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'DAYS_ID_PUBLISH': ['count', 'min', 'max', 'mean', 'median']})

# 'OWN_CAR_AGE': Age of client's car
df['OWN_CAR_AGE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df.groupby('TARGET').agg({'OWN_CAR_AGE': ['count', 'min', 'max', 'mean', 'median']}) # Little difference between labels.

# 'FLAG_MOBIL': Client's providing mobile phone information
# Did client provide mobile phone (1=YES, 0=NO)
# Percentage of customers who gave their mobile number: 0.99
# Number of customers providing the mobile phone: 307510
df['FLAG_MOBIL'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('FLAG_MOBIL').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_MOBIL', data=df)
plt.show()

# 'FLAG_EMP_PHONE': Client's providing work phone information
# Did client provide work phone (1=YES, 0=NO)
# Percentage of customers who gave their work phone: 0.81
# Number of customers providing the work phone: 252125
df['FLAG_EMP_PHONE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('FLAG_EMP_PHONE').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_EMP_PHONE', data=df)
plt.show()

# 'FLAG_WORK_PHONE': Client's providing work places phone information
# Did client provide work phone (1=YES, 0=NO)
# Percentage of customers who gave their work phone: 0.19
# Number of customers providing the work phone: 61308
df['FLAG_WORK_PHONE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('FLAG_WORK_PHONE').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_WORK_PHONE', data=df)
plt.show()

# 'FLAG_CONT_MOBILE': Availability of the Client's mobile phone
# Was mobile phone reachable (1=YES, 0=NO)
# Percentage of customers with mobile phone reachable: 0.99
# Number of customers with reachable mobile phones: 306937
df['FLAG_CONT_MOBILE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('FLAG_CONT_MOBILE').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_CONT_MOBILE', data=df)
plt.show()

# 'FLAG_EMAIL': Client's providing e-mail information
# Did client provide email (1=YES, 0=NO)
# percentage of customers who gave their email: 0.05
# number of customers providing the email: 17442
df['FLAG_EMAIL'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('FLAG_EMAIL').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_EMAIL', data=df)
plt.show()

# 'OCCUPATION_TYPE': What kind of occupation does the client have
# null: 31%, Laborers: 18%, Other (155934): 51%

# 'CNT_FAM_MEMBERS': How many family members does client have
# (min: 1 - max: 20)
# There are 17 different values.
# Those with more than 5 family members can be gathered in one class.
# new feature --> AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
df['CNT_FAM_MEMBERS'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('CNT_FAM_MEMBERS').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["CNT_FAM_MEMBERS"].hist(bins=20)
plt.xlabel("CNT_FAM_MEMBERS")
plt.title("CNT_FAM_MEMBERS")
plt.show()

# 'REGION_RATING_CLIENT': Our rating of the region where client lives (1,2,3)
# Target average for class 1: 0.04
# Target average for class 2: 0.07
# Target average for class 3: 0.11
# Class 3 has the highest risk rate
# The correlation between "REGION_RATING_CLIENT" and "REGION_RATING_CLIENT_W_CITY" is 0.95.
df['REGION_RATING_CLIENT'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('REGION_RATING_CLIENT').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REGION_RATING_CLIENT', data=df)
plt.show()

# 'REGION_RATING_CLIENT_W_CITY': Our rating of the region where client lives with taking city into account (1,2,3)
# Target average for class 1: 0.04
# Target average for class 2: 0.07
# Target average for class 3: 0.11
# Class 3 has the highest risk rate
# The correlation between "REGION_RATING_CLIENT" and "REGION_RATING_CLIENT_W_CITY" is 0.95.
# This variable can be dropped
df['REGION_RATING_CLIENT_W_CITY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('REGION_RATING_CLIENT_W_CITY').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REGION_RATING_CLIENT_W_CITY', data=df)
plt.show()

# 'WEEKDAY_APPR_PROCESS_START': On which day of the week did the client apply for the loan
# target averages are close to each other.
# minimum application is on Sunday
# max application --> TUESDAY 18%, WEDNESDAY 17%
df['WEEKDAY_APPR_PROCESS_START'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('WEEKDAY_APPR_PROCESS_START').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='WEEKDAY_APPR_PROCESS_START', data=df)
plt.show()

# 'HOUR_APPR_PROCESS_START': Approximately at what hour did the client apply for the loan
# application during working hours more (9 - 17)
# Applications can be divided into two classes as applications during working hours
# and applications outside of working hours.
df['HOUR_APPR_PROCESS_START'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('HOUR_APPR_PROCESS_START').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["HOUR_APPR_PROCESS_START"].hist(bins=20)
plt.xlabel("HOUR_APPR_PROCESS_START")
plt.title("HOUR_APPR_PROCESS_START")
plt.show()

# 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3': Normalized score from external data source.
#A number between 0 and 1.
#Corelations with TARGET
engineered_numerical_columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
X_eng = df[engineered_numerical_columns + ['TARGET']]
X_eng_corr = abs(X_eng.corr())
X_eng_corr.sort_values('TARGET', ascending=False)['TARGET']
# Describe for one of them
df['EXT_SOURCE_1'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable for one of them
df.groupby('EXT_SOURCE_1').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='WEEKDAY_APPR_PROCESS_START', data=df)
plt.show()

# 'ORGANIZATION_TYPE': Type of organization where client works.
df['ORGANIZATION_TYPE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['ORGANIZATION_TYPE'].value_counts()
df.groupby("ORGANIZATION_TYPE")["TARGET"].mean()
eda.cat_summary(df, ['ORGANIZATION_TYPE'], 'TARGET')
df.groupby('ORGANIZATION_TYPE').agg({'AMT_INCOME_TOTAL': ['count','mean', 'median']}).sort_values()
# Mean of Incomes with respect to REGION_RATING_CLIENT
sns.barplot(data=df, y='AMT_INCOME_TOTAL', x='ORGANIZATION_TYPE', )
# plt.xlabel(loc='left')
plt.show()
#Corelations with TARGET for 'AMT_INCOME_TOTAL'
#A new variable can be created for the organization type.
inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
inc_by_org
# Countplot
plt.figure(figsize=(12,5))
sns.countplot(y='ORGANIZATION_TYPE', data =df, hue='TARGET')
plt.show()

# 'REG_CITY_NOT_LIVE_CITY': Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)
df['REG_CITY_NOT_LIVE_CITY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['REG_CITY_NOT_LIVE_CITY'].value_counts()
# Group by target variable
df.groupby('REG_CITY_NOT_LIVE_CITY').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REG_CITY_NOT_LIVE_CITY', data=df)
plt.show()

# 'REG_CITY_NOT_WORK_CITY': Flag if client's permanent address does not match work address
# (1=different, 0=same, at city level)
df['REG_CITY_NOT_WORK_CITY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['REG_CITY_NOT_WORK_CITY'].value_counts()
# Group by target variable
df.groupby('REG_CITY_NOT_WORK_CITY').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REG_CITY_NOT_WORK_CITY', data=df)
plt.show()

# 'LIVE_CITY_NOT_WORK_CITY': Flag if client's contact address does not match work address
# (1=different, 0=same, at city level)
df['LIVE_CITY_NOT_WORK_CITY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['LIVE_CITY_NOT_WORK_CITY'].value_counts()
a = df.groupby('LIVE_CITY_NOT_WORK_CITY', as_index=False)['TARGET'].mean()
print(a)
# Group by target variable
df.groupby('LIVE_CITY_NOT_WORK_CITY').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='LIVE_CITY_NOT_WORK_CITY', data=df)
plt.show()


## LOCATION INFORMATION

# 'REG_REGION_NOT_LIVE_REGION': Flag if client's permanent address does not match contact address
# (1=different, 0=same, at region level)
# 0 (98,5%), 1 (1,5%)
df['REG_REGION_NOT_LIVE_REGION'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['REG_REGION_NOT_LIVE_REGION'].value_counts()
a = df.groupby('REG_REGION_NOT_LIVE_REGION', as_index=False)['TARGET'].mean()
print (a)
# Group by target variable
df.groupby('REG_REGION_NOT_LIVE_REGION').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REG_REGION_NOT_LIVE_REGION', data=df)
plt.show()

# 'REG_REGION_NOT_WORK_REGION': Flag if client's permanent address does not match work address
# (1=different, 0=same, at region level)
# 0 (95%), 1 (5%)
df['REG_REGION_NOT_WORK_REGION'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['REG_REGION_NOT_WORK_REGION'].value_counts()
a = df.groupby('REG_REGION_NOT_WORK_REGION', as_index=False)['TARGET'].mean()
print (a)
# Group by target variable
df.groupby('REG_REGION_NOT_WORK_REGION').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='REG_REGION_NOT_WORK_REGION', data=df)
plt.show()

# 'LIVE_REGION_NOT_WORK_REGION': Flag if client's contact address does not match work address
# (1=different, 0=same, at region level)
# 0 (95%), 1 (5%)
df['LIVE_REGION_NOT_WORK_REGION'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['LIVE_REGION_NOT_WORK_REGION'].value_counts()
a = df.groupby('LIVE_REGION_NOT_WORK_REGION', as_index=False)['TARGET'].mean()
print (a)
# Group by target variable
df.groupby('LIVE_REGION_NOT_WORK_REGION').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='LIVE_REGION_NOT_WORK_REGION', data=df)
plt.show()

## RESIDENCE STATISTICS

# 'APARTMENTS_AVG','APARTMENTS_MODE','APARTMENTS_MEDI': It is the total number of rooms of the apartment where the clients lives.
# It contains average, mode and median information, respectively.
# Describe
df['APARTMENTS_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('APARTMENTS_MEDI').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df['APARTMENTS_MEDI'].hist(bins=20)
plt.xlabel('APARTMENTS_MEDI')
plt.title('APARTMENTS_MEDI')
plt.show()

# 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 'BASEMENTAREA_MEDI': It is the information whether the apartment where the customer lives is a basement or not.
# It contains average, mode and median information, respectively.
df['BASEMENTAREA_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('BASEMENTAREA_AVG').agg({'TARGET': ['count', 'mean', 'median']})

# 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI': Year of building commissioning.
# It contains average, mode and median information, respectively.
df['YEARS_BEGINEXPLUATATION_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('YEARS_BEGINEXPLUATATION_MEDI').agg({'TARGET': ['count', 'mean', 'median']})

# 'YEARS_BUILD_AVG', 'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI': It is the year information of the flat where the customer lives.
# It contains average, mode and median information, respectively.
df['YEARS_BUILD_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# 'COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI': It is the common area information of the apartment where the customer lives.
# It contains average, mode and median information, respectively.
df['COMMONAREA_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# 'ELEVATORS_AVG', 'ELEVATORS_MODE', 'ELEVATORS_MEDI': It is the elevator information in the apartment where the customer lives.
# It contains average, mode and median information, respectively.
df['ELEVATORS_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('ELEVATORS_MEDI').agg({'TARGET': ['count', 'mean', 'median']})

# 'ENTRANCES_AVG', 'ENTRANCES_MODE', 'ENTRANCES_MEDI': It is the entrances information in the place where the customer lives.
# It contains average, mode and median information, respectively.
df['ENTRANCES_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('ENTRANCES_MEDI').agg({'TARGET': ['count', 'mean', 'median']})

# 'FLOORSMAX_AVG', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI': It is the information about the maximum number of floors of the apartment where the customer lives.
# It contains average, mode and median information, respectively.
df['FLOORSMAX_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('FLOORSMAX_MEDI').agg({'TARGET': ['count', 'mean', 'median']})

# 'FLOORSMIN_AVG', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI': It is the information about the minimum number of floors of the apartment where the customer lives.
# It contains average, mode and median information, respectively.
df['FLOORSMIN_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('FLOORSMIN_AVG').agg({'TARGET': ['count', 'mean', 'median']})

# 'LANDAREA_AVG', 'LANDAREA_MODE', 'LANDAREA_MEDI': It is the land area information of the apartment where the customer lives.
# It contains average, mode and median information, respectively.
df['LANDAREA_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('LANDAREA_MEDI').agg({'TARGET': ['count', 'mean', 'median']})

# 'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI': It is the information about the number of livable rooms of the customer.
# It contains average, mode and median information, respectively.
df['LIVINGAPARTMENTS_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('LIVINGAPARTMENTS_AVG').agg({'TARGET': ['count', 'mean', 'median']})

# 'LIVINGAREA_AVG', 'LIVINGAREA_MODE', 'LIVINGAREA_MEDI': It is the customer's livable area at home
# It contains average, mode and median information, respectively.
df['LIVINGAREA_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('LIVINGAREA_AVG').agg({'TARGET': ['count', 'mean', 'median']})

# 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_MEDI': It is the room information that is not used in the client's home
# It contains average, mode and median information, respectively.
df['NONLIVINGAPARTMENTS_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.groupby('NONLIVINGAPARTMENTS_AVG').agg({'TARGET': ['count', 'mean', 'median']})

# 'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_MEDI': It is the information of living space that is not used in the client's home.
# It contains average, mode and median information, respectively.
df['NONLIVINGAREA_MEDI'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# 'FONDKAPREMONT_MODE': It is the donation fund given by the customer for the home.
#how building repair/renewal is financed.
df['FONDKAPREMONT_MODE'].value_counts()
eda.cat_summary(df, ['FONDKAPREMONT_MODE'], 'TARGET')
sns.barplot(data=df, y='TARGET', x='ORGANIZATION_TYPE', )
plt.show()
# reg oper account         85954   : Common regional fund, supposedly better for older buildings
# reg oper spec account    14070
# not specified             6600
# org spec account          6539
# One of: nan 210295 (68.39%),
# not specified 5687 (1.85%),
# org spec account 5619 (1.83%),
# reg oper account 73830 (24.01%),
# reg oper spec account 12080 (3.93%).

# 'HOUSETYPE_MODE': Provides information about the structure of the client house.
df['HOUSETYPE_MODE'].value_counts()
eda.cat_summary(df, ['HOUSETYPE_MODE'], 'TARGET')
# One of block of flats 150503 (48.94%),
# nan 154297 (50.18%),
# specific housing 1499 (0.49%),
# terraced house 1212 (0.39%).

# 'TOTALAREA_MODE': It is the total area information of the house.
df['TOTALAREA_MODE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['TOTALAREA_MODE'].value_counts()
eda.cat_summary(df, ['TOTALAREA_MODE'], 'TARGET')

# 'WALLSMATERIAL_MODE': It is the wall material information of the client's house
df['WALLSMATERIAL_MODE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['WALLSMATERIAL_MODE'].value_counts()
eda.cat_summary(df, ['WALLSMATERIAL_MODE'], 'TARGET')
# One of Block 9253 (3.01%),
# Mixed 2296 (0.75%),
# Monolithic 1779 (0.58%),
# Others 1625 (0.53%),
# Panel 66040 (21.48%),
# Stone, brick 64815 (21.08%),
# Wooden 5362 (1.74%),
# nan 156341 (50.84%)

# 'EMERGENCYSTATE_MODE': It is the eligibility / qualification status of the house.
df['EMERGENCYSTATE_MODE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df['EMERGENCYSTATE_MODE'].value_counts()
eda.cat_summary(df, ['EMERGENCYSTATE_MODE'], 'TARGET')
a = df.groupby('EMERGENCYSTATE_MODE', as_index=False)['TARGET'].mean()
print (a)
(df['EMERGENCYSTATE_MODE'].value_counts() / len(df)) * 100
# No 159428 (% 51,84),
# Yes 2328 (% 0,76),
# nan 145755 (% 47,40)

# 'OBS_30_CNT_SOCIAL_CIRCLE': How many observation of client's social surroundings with observable 30 DPD (days past due) default
# It includes the information about how many people are in the social circle of the customer.
# Those who took a loan at least 30 days before these people are included in this variable.
# The correlation between "OBS_30_CNT_SOCIAL_CIRCLE" and "OBS_60_CNT_SOCIAL_CIRCLE" is 1.
# There is no one in the social circle of 163910 customers.
# 142580 customers have at least one person in their social circle.
df['OBS_30_CNT_SOCIAL_CIRCLE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('OBS_30_CNT_SOCIAL_CIRCLE').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["OBS_30_CNT_SOCIAL_CIRCLE"].hist(bins=20)
plt.xlabel("OBS_30_CNT_SOCIAL_CIRCLE")
plt.title("OBS_30_CNT_SOCIAL_CIRCLE")
plt.show()

# 'DEF_30_CNT_SOCIAL_CIRCLE': How many observation of client's social surroundings defaulted on 30 DPD (days past due)
# The information on how many people from the social circle of the client goes into default is kept in this variable.
# The correlation between "DEF_30_CNT_SOCIAL_CIRCLE" and "DEF_60_CNT_SOCIAL_CIRCLE" is 0.86.
df['DEF_30_CNT_SOCIAL_CIRCLE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('DEF_30_CNT_SOCIAL_CIRCLE').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["DEF_30_CNT_SOCIAL_CIRCLE"].hist(bins=20)
plt.xlabel("DEF_30_CNT_SOCIAL_CIRCLE")
plt.title("DEF_30_CNT_SOCIAL_CIRCLE")
plt.show()

# 'OBS_60_CNT_SOCIAL_CIRCLE': How many observation of client's social surroundings with observable 60 DPD (days past due) default
# The correlation between "OBS_30_CNT_SOCIAL_CIRCLE" and "OBS_60_CNT_SOCIAL_CIRCLE" is 1.
# This variable can be dropped
df['OBS_60_CNT_SOCIAL_CIRCLE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('OBS_60_CNT_SOCIAL_CIRCLE').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["OBS_60_CNT_SOCIAL_CIRCLE"].hist(bins=20)
plt.xlabel("OBS_60_CNT_SOCIAL_CIRCLE")
plt.title("OBS_60_CNT_SOCIAL_CIRCLE")
plt.show()

# 'DEF_60_CNT_SOCIAL_CIRCLE': How many observation of client's social surroundings defaulted on 60 (days past due) DPD
# The correlation between "DEF_30_CNT_SOCIAL_CIRCLE" and "DEF_60_CNT_SOCIAL_CIRCLE" is 0.86.
# this variable can be dropped
df['DEF_60_CNT_SOCIAL_CIRCLE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('DEF_60_CNT_SOCIAL_CIRCLE').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["DEF_60_CNT_SOCIAL_CIRCLE"].hist(bins=20)
plt.xlabel("DEF_60_CNT_SOCIAL_CIRCLE")
plt.title("DEF_60_CNT_SOCIAL_CIRCLE")
plt.show()

# 'DAYS_LAST_PHONE_CHANGE': How many days before application did client change phone
# (min: -4292.0 - max: 0)
# Number of customers who did not change their phone number: 37672
# Days can be changed to years
df['DAYS_LAST_PHONE_CHANGE'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('DAYS_LAST_PHONE_CHANGE').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["DAYS_LAST_PHONE_CHANGE"].hist(bins=20)
plt.xlabel("DAYS_LAST_PHONE_CHANGE")
plt.title("DAYS_LAST_PHONE_CHANGE")
plt.show()

# "FLAG_DOCUMENT_2"  <--->  "FLAG_DOCUMENT_21"  (20 documents): Did the customer provide the required documents?
# 0: document provided
# 1: document not provided
# these variables have no null values
# All variables containing document information can be collected.
df.groupby('FLAG_DOCUMENT_3').agg({'TARGET': ['count', 'mean', 'median']})
# Countplot
sns.countplot(x='FLAG_DOCUMENT_3', data=df)
plt.show()

## Variables containing the number of queries to Credit Bureau about the customer.
# These variables can be summed

# 'AMT_REQ_CREDIT_BUREAU_HOUR': Number of enquiries to Credit Bureau about the client one hour before application
# There are 5 different values in the variable (0, 1, 2, 3, 4)
# 1626 people in total
df['AMT_REQ_CREDIT_BUREAU_HOUR'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_HOUR').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_HOUR"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_HOUR")
plt.title("AMT_REQ_CREDIT_BUREAU_HOUR")
plt.show()

# 'AMT_REQ_CREDIT_BUREAU_DAY': Number of enquiries to Credit Bureau about the client one day before application
# (excluding one hour before application)
# The number of queries made to the Credit Bureau about the client
# one hour before the application is included in the "AMT_REQ_CREDIT_BUREAU_HOUR" variable.
# There are 9 different values in the variable (0 to 8)
# 1489 people in total
df['AMT_REQ_CREDIT_BUREAU_DAY'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_DAY').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_DAY"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_DAY")
plt.title("AMT_REQ_CREDIT_BUREAU_DAY")
plt.show()

# 'AMT_REQ_CREDIT_BUREAU_WEEK': Number of enquiries to Credit Bureau about the client one week before application
# (Excluding one day before application)
# The number of queries made to the Credit Bureau about the client
# one day before the application is included in the "AMT_REQ_CREDIT_BUREAU_DAY" variable.
# There are 9 different values in the variable (0 to 8)
# 8536 people in total
df['AMT_REQ_CREDIT_BUREAU_WEEK'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_WEEK').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_WEEK"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_WEEK")
plt.title("AMT_REQ_CREDIT_BUREAU_WEEK")
plt.show()

# 'AMT_REQ_CREDIT_BUREAU_MON': Number of enquiries to Credit Bureau about the client one month before application
# (Excluding one week before application)
# The number of queries made to the Credit Bureau about the client
# one week before the application is included in the "AMT_REQ_CREDIT_BUREAU_WEEK" variable.
# There are 24 different values in the variable (min: 0 - max: 27)
# 43759 people in total
# Three biggest values (Except 0):
# 1.0      33147
# 2.0       5386
# 3.0       1991
df['AMT_REQ_CREDIT_BUREAU_MON'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_MON').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_MON"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_MON")
plt.title("AMT_REQ_CREDIT_BUREAU_MON")
plt.show()

# 'AMT_REQ_CREDIT_BUREAU_QRT': Number of enquiries to Credit Bureau about the client 3 month before application
# (Excluding one month before application)
# The number of queries made to the Credit Bureau about the client
# one month before the application is included in the "AMT_REQ_CREDIT_BUREAU_MON" variable.
# There are 11 different values in the variable (min: 0 - max: 261) ---> anomaly
# 50575 people in total
# Three biggest values (Except 0):
# 1.0       33862
# 2.0       14412
# 3.0        1717
df['AMT_REQ_CREDIT_BUREAU_QRT'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_QRT').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_QRT"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_QRT")
plt.title("AMT_REQ_CREDIT_BUREAU_QRT")
plt.show()

# 'AMT_REQ_CREDIT_BUREAU_YEAR': Number of enquiries to Credit Bureau about the client one day year
# (Excluding last 3 months before application)
# The number of queries made to the Credit Bureau about the client
# 3 month before the application is included in the "AMT_REQ_CREDIT_BUREAU_QRT" variable.
# There are 25 different values in the variable (min: 0 - max: 25)
# 194191 people in total
# Three biggest values (Except 0):
# 1.0     63405
# 2.0     50192
# 3.0     33628
df['AMT_REQ_CREDIT_BUREAU_YEAR'].describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
# Group by target variable
df.groupby('AMT_REQ_CREDIT_BUREAU_YEAR').agg({'TARGET': ['count', 'mean', 'median']})
# Histogram
df["AMT_REQ_CREDIT_BUREAU_YEAR"].hist(bins=20)
plt.xlabel("AMT_REQ_CREDIT_BUREAU_YEAR")
plt.title("AMT_REQ_CREDIT_BUREAU_YEAR")
plt.show()





