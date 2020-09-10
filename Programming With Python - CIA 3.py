#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on L&T Car Loan Default Data

# ## Industry Profile:
# 
#    The car loan market in India has been growing at a quick pace for the last decade and is
# projected to do so until 2025. Aggressive competitive pricing, high replacement demand and
# government incentives to shift to electric vehicles are among the major factors that are driving
# this growth. Major car manufacturing companies of the world are setting up plants in India. In
# 2017, EY projected that the car loans market would double in the next 3 years. In 2018 India
# witnessed a whopping 3.3 million in sales of passenger cars. This was the 3rd highest in volume
# only behind Germany and Japan. This resulted in a much larger demand in the car loans market.
# Post that point, due to the shift from BS IV to BS VI(increase in prices), and now because of
# COVID 19 the sales have dropped considerably. Hence the car loans market has also shrunk
# because of the same. In 2019, the market shrank from 22.5 thousand crores to 20.5 thousand
# crores. In 2020, the number plummeted further.
# 
#    The car loans market can be segmented on the type of car, percentage of amount disbursed,
# source, type of city, tenure of the loan etc. Under car types we have SUVs, hatchbacks and sedans.
# Sedans are predominantly the most selling type of cars in India due to several factors like riding
# comfort, safety, status associated with a sedan. The government has offered a tax cut up to 12%
# on the purchase of electric vehicles. The benefits also extend to loans where they have a made a
# provision of 1.5 lakh rupees in tax benefits on loans taken out for the purchase of electric cars.
# Hence, the loans market stands to benefit from both of these provisions.
# 
#    The institutions that are offering car loans on purchase of cars in India can be classified
# into banks, OEMs or non-banking financial corporations (NBFCs). Original Equipment
# manufacturers are coming up with several provisions for credit to attract more customers. Loans
# are even personalized based on individual requirement to tailor the product to each customer.
# Even though banks take up the lion's share in the Indian car loans market (due to their large
# customer base) the market share of OEMs and NBFCs is increasing by the day. One such
# institution is L&T Finance.

# ## Company Profile: L&T Finance
# 
#    L&T Finance Holdings Limited is a non-banking financial institution-core investment
# company. The Company's segments include Retail and Mid-Market Finance, which consists of
# rural products finance, personal vehicle finance, microfinance, housing finance, commercial
# vehicle finance, construction equipment finance, loans and leases and loan against shares;
# Wholesale Finance, which consists of project finance and non-project corporate finance to infra
# and non-infra segments across power-thermal and renewable; transportation-roads, ports and
# airports; telecom, and other non-infra segments; Investment Management, which consists of
# assets management of mutual fund and private equity fund, and Other Business, which consists
# of wealth management and financial product distribution. It offers a range of financial products
# and services across retail, corporate, housing and infrastructure finance sectors, as well as mutual
# fund products and investment management services.
# 
#    The data considered for the project was taken from Kaggle – An open source platform for
# data. An attempt has been made to create models both for policy decisions and for loan approval
# decisions. The data is exclusive to the loans given to sales of cars, that is, SUVs, hatchbacks and
# sedans only. The tractor loans which is also a major part of L&T finance’s vehicular loan segment
# is not included in the data.

# ## Data Understanding
# 
# The data set considered is https://www.kaggle.com/mamtadhaker/lt-vehicle-loandefault-prediction. It is data from an open source platform called Kaggle, it comprises
# of 2,33,154 rows and 41 columns. The data dictionary with the descriptions of all the
# variables is given below.

# ### Here is a list of all the columns in the data set and their meanings. 
# 
# UniqueID	Identifier for customers
# 
# loan_default	Payment default in the first EMI on due date
# 
# disbursed_amount	Amount of Loan disbursed
# 
# asset_cost	Cost of the Asset
# 
# ltv	Loan to Value of the asset
# 
# branch_id	Branch where the loan was disbursed
# 
# supplier_id	Vehicle Dealer where the loan was disbursed
# 
# manufacturer_id	Vehicle manufacturer(Hero, Honda, TVS etc.)
# 
# Current_pincode	Current pincode of the customer
# 
# Date.of.Birth	Date of birth of the customer
# 
# Employment.Type	Employment Type of the customer (Salaried/Self Employed)
# 
# DisbursalDate	Date of disbursement
# 
# State_ID	State of disbursement
# 
# Employee_code_ID	Employee of the organization who logged the disbursement
# 
# MobileNo_Avl_Flag	if Mobile no. was shared by the customer then flagged as 1
# 
# Aadhar_flag	if aadhar was shared by the customer then flagged as 1
# 
# PAN_flag	if pan was shared by the customer then flagged as 1
# 
# VoterID_flag	if voter  was shared by the customer then flagged as 1
# 
# Driving_flag	if DL was shared by the customer then flagged as 1
# 
# Passport_flag	if passport was shared by the customer then flagged as 1
# 
# PERFORM_CNS.SCORE	Bureau Score
# 
# PERFORM_CNS.SCORE.DESCRIPTION	Bureau score description
# 
# PRI.NO.OF.ACCTS	count of total loans taken by the customer at the time of disbursement
# 
# PRI.ACTIVE.ACCTS	count of active loans taken by the customer at the time of disbursement
# 
# PRI.OVERDUE.ACCTS	count of default accounts at the time of disbursement
# 
# PRI.CURRENT.BALANCE	total Principal outstanding amount of the active loans at the time of disbursement
# 
# PRI.SANCTIONED.AMOUNT	total amount that was sanctioned for all the loans at the time of disbursement
# 
# PRI.DISBURSED.AMOUNT	total amount that was disbursed for all the loans at the time of disbursement
# 
# SEC.NO.OF.ACCTS	count of total loans taken by the customer at the time of disbursement
# 
# SEC.ACTIVE.ACCTS	count of active loans taken by the customer at the time of disbursement
# 
# SEC.OVERDUE.ACCTS	count of default accounts at the time of disbursement
# 
# SEC.CURRENT.BALANCE	total Principal outstanding amount of the active loans at the time of disbursement
# 
# SEC.SANCTIONED.AMOUNT	total amount that was sanctioned for all the loans at the time of disbursement
# 
# SEC.DISBURSED.AMOUNT	total amount that was disbursed for all the loans at the time of disbursement
# 
# PRIMARY.INSTAL.AMT	EMI Amount of the primary loan
# 
# SEC.INSTAL.AMT	EMI Amount of the secondary loan
# 
# NEW.ACCTS.IN.LAST.SIX.MONTHS	New loans taken by the customer in last 6 months before the disbursment
# 
# DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS	Loans defaulted in the last 6 months
# 
# AVERAGE.ACCT.AGE	Average loan tenure
# 
# CREDIT.HISTORY.LENGTH	Time since first loan
# 
# NO.OF_INQUIRIES	Enquries done by the customer for loans

# Let us load all the necessary Libraries to do exploratory data analysis.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
# Instructions to ignore warnings and other miscellaneous tweeks
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')


# Then we load the Data into python. 

# In[2]:


data = pd.read_csv('train.csv')


# Let has have a look at the data, the data seems well structured. 

# In[3]:


data.head()


# Let us find the dimentions of the data 

# In[4]:


data.shape


# To list all the Columns Headings in the data 

# In[5]:


data.columns


# Let us see if there are any missing values in the data set. The following code will tell us how many null values are there in each column. 

# In[6]:


data.isnull().sum() 


# To find the percentage of missing values for the column - Employment.Type as that is the only column that has missing values. 

# In[7]:


print('Percentage of missing values is {0}%'.format(round(100*data['Employment.Type'].isnull().sum()/len(data),3)))


# We can consider eliminating the same, but there is a better way. We can create a level called - unknown in the employment column. That will help us retain all available data for analysis. 
# First, let us find out how many unique entries are there in each field.

# In[8]:


data.apply(lambda x: len(x.unique()))


# In[9]:


# We see that there are only 3 unique entries in Employee.Type. Let us see what they are.  
data['Employment.Type'].value_counts()


# The third unique kind is invariably the missing values. Now, let us look at creation of an additional factor for Employment.Type Column called 'unknown' to replace all the missing entries in the column.

# In[10]:


data.fillna('unknown', inplace=True)
data['Employment.Type'].value_counts()


# Now we have replaced all NA's by 'unknown' in the column. This means that the data is free of Missing values. 
# Let us look at the basic statistical information of the data. 

# In[11]:


data.describe()


# In[12]:


# Now let us see the class distribution of the output variable 
data.loan_default.value_counts()


# In[13]:


# Visualising the same, 
data.loan_default.value_counts().plot(kind = 'bar')


# There is a clear Class Imbalance. There are more non-defaulters than defaulters. Which is as expected. Before we build a model, we must therefore use upsampling or downsampling to deal with the class imbalance. 
# Now, let us look at all the column headings. 

# In[14]:


data.columns


# Let us drop all the columns that will not contribute to data analysis. 

# In[15]:


data=data.drop(['UniqueID', 'branch_id','supplier_id', 'Current_pincode_ID','State_ID', 'Employee_code_ID', 'MobileNo_Avl_Flag'],axis=1)


# Here we are definig a function to create an additional factor called unknown to replace all '-' values in the credit risk column 
# Then, we are converting categorical variables into numerical variables. 

# In[16]:


def credit_risk(data):#to replace all '-' entiries in the column to 'unknown'
    d1=[]
    d2=[]
    for i in data:
        p = i.split("-")
        if len(p) == 1:
            d1.append(p[0])
            d2.append('unknown')
        else:
            d1.append(p[1])
            d2.append(p[0])

    return d2

sub_risk = {'unknown':-1, 'A':13, 'B':12, 'C':11,'D':10,'E':9,'F':8,'G':7,'H':6,'I':5,'J':4,'K':3, 'L':2,'M':1}
employment_map = {'Self employed':0, 'Salaried':1, 'unknown':2}

data.loc[:,'credit_risk_grade']  = credit_risk(data["PERFORM_CNS.SCORE.DESCRIPTION"])
data.loc[:,'Credit Risk'] = data['credit_risk_grade'].apply(lambda x: sub_risk[x])

data.loc[:,'Employment Type'] = data['Employment.Type'].apply(lambda x: employment_map[x])

data=data.drop(['PERFORM_CNS.SCORE.DESCRIPTION', 'credit_risk_grade','Employment.Type'],axis=1)


# Conversion of date variables into age. 

# In[17]:


def age(dur):#creating a function to convert date to age. 
    yr = int(dur.split('-')[2])
    if yr >=0 and yr<=20:
        return yr+2000
    else:
         return yr+1900

data['Date.of.Birth'] = data['Date.of.Birth'].apply(age)
data['DisbursalDate'] = data['DisbursalDate'].apply(age)
data['Age']=data['DisbursalDate']-data['Date.of.Birth']
data=data.drop(['DisbursalDate','Date.of.Birth'],axis=1)


# Let us create a heat map to see correlation of various columns in the given data set. 

# In[18]:


plt.figure(figsize=(30,20))
sns.heatmap(data.corr())
plt.show()


# Now let us see the correlation values in a table. 

# In[19]:


data.corr()


# In[20]:


df1=data[data['loan_default']==1]
df0=data[data['loan_default']==0]


# This shows the distribution of disbursed amount for all loans. 

# In[21]:


sns.boxplot(data['disbursed_amount'])


# The following graph will compare disbursed amount for defaulters and non defaulters

# In[22]:


plt.figure(figsize=(10,4))
sns.distplot(df0['disbursed_amount'],kde=False)
sns.distplot(df1['disbursed_amount'],kde=False)
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# Let us apply logarithmic transformation to see the distribution better.

# In[23]:


sns.distplot(data['disbursed_amount'].apply(lambda x:np.log(x)))


# Let us now consider the loan to asset ratio.

# In[24]:


sns.boxplot(data['ltv'])


# In[25]:


sns.distplot(data['ltv'])


# Let us see defaulters and non defaulters separately

# In[26]:


plt.figure(figsize=(10,4))
sns.distplot(df0['ltv'],kde=False)
sns.distplot(df1['ltv'],kde=False)
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# From the above diagram it is clear that defaulters have a lower loan to asset(value) ratio. 
# Let us clearly separate numerical and categorical variables. 

# In[27]:


numerical=['disbursed_amount','asset_cost','PRI.NO.OF.ACCTS',
       'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE',
       'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS',
       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',
       'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
       'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES','Age','NEW.ACCTS.IN.LAST.SIX.MONTHS', 
        'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']
categorical=['manufacturer_id', 'Aadhar_flag', 'PAN_flag',
       'VoterID_flag', 'Driving_flag', 'Passport_flag', 'PERFORM_CNS.SCORE',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
       'AVERAGE.ACCT.AGE', 'NO.OF_INQUIRIES', 'Credit Risk','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH',
       'Employment Type']


# T-Test for numerical columns, visualised as a bar graph.

# In[28]:


p=[]
from scipy.stats import ttest_ind

for i in numerical:
    data1=data.groupby('loan_default').get_group(0)
    data2=data.groupby('loan_default').get_group(1)
    t,pvalue=ttest_ind(data1[i],data2[i])
    p.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=p, y=numerical)
plt.title('Best Numerical Features')
plt.axvline(x=(1-0.05),color='r')
plt.xlabel('1-p value')
plt.show()


# The Graph shows 'Acceptance of Alternate Hypothesis'. The features that cross the red line show statistical signifance as per T-test. The two features that do not cross the line are both secondary account features, which banks by convention do not ignore(being a guarantor for someone else's loan). So let us keep them and go ahead. T tests are conducted assuming the standard deviations of any two samples taken from the population will be the same. Let us check that for the given Dataset to ensure the credibility of T test.

# In[29]:


for i in numerical:
    data1=data.groupby('loan_default').get_group(0)
    data2=data.groupby('loan_default').get_group(1)
    print(np.std(data1[i],ddof=1),np.std(data2[i],ddof=1))


# Here we have taken 2 samples. One sample of defaulters and the other for non defaulters. The display here compares the standard deviation for every column in the two samples. There are differences between standard deviations. Even though the differences seem not too big, it is still unacceptable. Let us go ahead with a non parametric test.

# In[30]:


# let us use SelectKBest library to narrow down choices of features. This will make use of Annova test.
# the figure shows which factors might drive loan default
from sklearn.feature_selection import SelectKBest,f_classif
n = SelectKBest(score_func=f_classif, k=10)
numcols=n.fit(data[numerical],data['loan_default'])
plt.figure(figsize=(7,7))
sns.barplot(x=numcols.scores_,y=numerical)
plt.title('Best Numerical Features')
plt.show()


# The secondary account details seem to have insignificant effect. We cannot afford to drop them because of convention. Let us combine them with Primary account features instead. This will work in our favour and it decreases the number of features.

# In[31]:


data.loc[:,'No of Accounts'] = data['PRI.NO.OF.ACCTS'] + data['SEC.NO.OF.ACCTS']
data.loc[:,'PRI Inactive accounts'] = data['PRI.NO.OF.ACCTS'] - data['PRI.ACTIVE.ACCTS']
data.loc[:,'SEC Inactive accounts'] = data['SEC.NO.OF.ACCTS'] - data['SEC.ACTIVE.ACCTS']
data.loc[:,'Total Inactive accounts'] = data['PRI Inactive accounts'] + data['SEC Inactive accounts']
data.loc[:,'Total Overdue Accounts'] = data['PRI.OVERDUE.ACCTS'] + data['SEC.OVERDUE.ACCTS']
data.loc[:,'Total Current Balance'] = data['PRI.CURRENT.BALANCE'] + data['SEC.CURRENT.BALANCE']
data.loc[:,'Total Sanctioned Amount'] = data['PRI.SANCTIONED.AMOUNT'] + data['SEC.SANCTIONED.AMOUNT']
data.loc[:,'Total Disbursed Amount'] = data['PRI.DISBURSED.AMOUNT'] + data['SEC.DISBURSED.AMOUNT']
data.loc[:,'Total Installment'] = data['PRIMARY.INSTAL.AMT'] + data['SEC.INSTAL.AMT']


# Let us conduct a similar test for categorical variables - Chi Square test

# In[32]:


from scipy.stats import chi2_contingency
l=[]
for i in categorical:
    pvalue  = chi2_contingency(pd.crosstab(data['loan_default'],data[i]))[1]
    l.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=l, y=categorical)
plt.title('Best Categorical Features')
plt.axvline(x=(1-0.05),color='r')
plt.show()


# Like the previous graph, this one too shows 'Acceptance of Alternate Hypothesis'. Here every feature seems significant except for PAN_flag. PAN Card is mandatory to obtain credit score. Hence we can observe some correlation(multicolleniarity) between credit score and PAN Card Because of which the Chi Square test deemed it surplus to our model.

# In[33]:


def duration(dur):#to find out for how long the loan has been active
    yrs = int(dur.split(' ')[0].replace('yrs',''))
    mon = int(dur.split(' ')[1].replace('mon',''))
    return yrs*12+mon


# In[34]:


data['AVERAGE.ACCT.AGE'] = data['AVERAGE.ACCT.AGE'].apply(duration)
data['CREDIT.HISTORY.LENGTH'] = data['CREDIT.HISTORY.LENGTH'].apply(duration)


# Let us drop all the columns we don't need anymore.

# In[35]:


data=data.drop(['PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','PRI.CURRENT.BALANCE','PRI Inactive accounts','SEC Inactive accounts',
            'PRI.SANCTIONED.AMOUNT','SEC.NO.OF.ACCTS','PRI.NO.OF.ACCTS','PRI.DISBURSED.AMOUNT','PRI.ACTIVE.ACCTS', 
            'PRI.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT', 'SEC.OVERDUE.ACCTS',
            'SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','disbursed_amount','SEC.ACTIVE.ACCTS'],axis=1)


# In[36]:


nums=['asset_cost', 'ltv','PERFORM_CNS.SCORE',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES','No of Accounts', 'Total Inactive accounts',
       'Total Overdue Accounts', 'Total Current Balance', 'Total Sanctioned Amount',
       'Total Disbursed Amount', 'Total Installment','Age']


# To find the number of columns in the final transformed, clean data.

# In[37]:


len(nums)


# Basic Model Building

# In[38]:


y=data.loan_default
X=data.drop("loan_default",axis=1)
from sklearn.model_selection import train_test_split,KFold,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print('train accuracy :',lr.score(X_train,y_train))
print('test accuracy :',lr.score(X_test,y_test))
print("precision :",precision_score(y_test,y_pred),"\n")
print("recall :",recall_score(y_test,y_pred),"\n")
print("f1 score:",f1_score(y_test,y_pred),"\n")
print(classification_report(y_test,y_pred))


# Even though statistically this model has an accuracy of 78%, it useless from the business perspective. We see that the Recall score of the model is 0. Which means that the model completely fails to predict the defaulters. To get better results we have to deal with the Class Imbalance.

# In[39]:


#Let us have a look at how the Credit History features are distributed for the given data. 
n=['PERFORM_CNS.SCORE','NO.OF_INQUIRIES','No of Accounts', 'Total Inactive accounts',
       'Total Overdue Accounts', 'Total Current Balance', 'Total Sanctioned Amount',
       'Total Disbursed Amount', 'Total Installment']
df=data[n]
fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(20,10))
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Distributions of Credit History Features')

for ax, feature, name in zip(axes.flatten(), data.values.T, data.columns):
    sns.distplot(feature, ax=ax)
    ax.set(title=str(name))
plt.show()


# There are a large number of '0's in the above data. That is because majority of the people are availing loans for the first time. Which is a common for real world loan applicants. We also see that there are some outliers which will complicate the process further. Instead of trying to remove the outliers, we can try to bring them closer to the median values.
Further steps will involve model building. That does not fall under the scope of the assignment. Hence I will stop the EDA with Data Cleaning here. 