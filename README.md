# Credit Risk Analysis

## Background
Credit risk analysis is a crucial process in the financial industry that involves assessing the likelihood of a borrower defaulting on their loan. Inaccurate credit risk assessments can result in losses, and bankruptcy as evident during the global financial crisis of 2008. Machine learning and big data analysis has emerged as powerful tools for analysing credit risk, enabling lenders to make more informed lending decisions.

## Problem Statement
The objective is to identify an accurate predictive model that can effectively evaluate the likelihood of loan defaults by new customers. This will enable lenders worldwide to make well-informed decisions in customer selection and loaning, thereby minimizing potential losses.

## Dataset
The dataset is a collection of payment information of 30,000 credit card holders from a bank in Taiwan https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients (obtained from UCI repository).  The data consists of information such as default payments, the demographic of the credit card holders, payment history and bill statements from April 2005 to September 2005. The breakdown of the 25 feature attributes is as follows: 

**Information of the customers:**
- `ID`: ID of each customer
- `X1 LIMIT_BAL`: Amount of given credit in NT dollar for both the individual customer and their family
- `X2 SEX`: Gender (1 = male; 2 = female)
- `X3 EDUCATION`: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others, 5 = unknown, 6 = unknown)
- `X4 MARRIAGE`: Marital status (1 = married; 2 = single; 3 = others)
- `X5 AGE`: Age of the customer

**Other information of the dataset also includes customer's delay of past payment, amount of bill statement and amount of previous statement**

## Pre-Processing of dataset
After loading the dataset we proceeded with the following:
- drop `NaN` values
- remove duplicate records
- identifying outliers
- checking for correct column types
- checking for categorical data

```
#removing Null values
```

## Explorator Data Analysis




