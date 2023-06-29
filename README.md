# Credit Risk Analysis
- Full report can be accessed [here](https://github.com/thisissteff/credit_risk_analysis/blob/main/creditRiskModelling.pdf)
- Full R Markdown file can be accessed [here](https://github.com/thisissteff/credit_risk_analysis/blob/main/Grp43RCode.pdf)

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
- check data types
- check and rename for consistent column naming
- check and remove `NaN` values
- check and remove duplicate records
- check for outliers in `X1 LIMIT_BAL`
- check for unrealistic ages

```
#rename column
colnames(credit)[colnames(credit) == "PAY_0"] <- "PAY_1"
colnames(credit)[colnames(credit) == "default.payment.next.month"] <- "DEFAULT"
# check for NaN values
missing_values <- colSums(is.na(credit))
# remove duplicate
credit.dup <- credit %>%
  select(-ID) # remove ID column since it is not needed to check for duplicates
duplicates <- duplicated(credit.dup) # creates a logical vector indicating duplicated rows
credit <- credit[!duplicates, ]
```

## Feature Selection
We select features that contribute most to our predict variable `DEFAULT`. We drop the variable `ID` as it does not help us in predicting the variable to be predicted, `DEFAULT`. We also drop the variable `DEFAULT` as this is the variable we will be predicting. After plotting out the correlation matrix, we decided to include all the other variables to help us in the prediction. 
<img width="808" alt="Screenshot 2023-06-29 at 4 47 49 PM" src="https://github.com/thisissteff/credit_risk_analysis/assets/138084370/79bab0c5-4fa6-4c82-ada9-6c33208d3690">


## Exploratory Data Analysis
We observed that out of 29965 data points, there are 23373 customers (78%) with non-default payments and 6592 customers (22%) with default. However, this is not a cause for concern as we would expect fewer defaults as compared to non-default payments.

<img width="535" alt="Screenshot 2023-06-29 at 4 56 30 PM" src="https://github.com/thisissteff/credit_risk_analysis/assets/138084370/988025b6-3659-4f54-a49c-072bc58c31b8">



## Model Training
We generated a confusion matrix for each model that will show the number of outcomes predicted by the models. In this analysis, we will treat the default case with a value of 1 as the positive outcome.

Splitting data into training and testing sets: 
```
set.seed(123)
train_index <- sample(1:nrow(credit), size = round(0.7*nrow(credit)), replace = FALSE)
train_data <- credit[train_index,]
test_data <- credit[-train_index,]
```

#### Simple Logistic Regression - Generalised Linear Model (GLM)

Training the model: 
```
# Convert DEFAULT to factor with levels "No default" and "Default"
credit$DEFAULT <- factor(credit$DEFAULT, levels = c(0, 1), labels = c("No default", "Defau

# Train logistic regression model
glm_model <- glm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + age_bins +
             PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
             PAY_AMT1+ PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6 +
             BILL_AMT1  + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6,
             family = binomial(link = "logit"), data = train_data)
```
Calculating the confusion matrix: 

```
# Make predictions on test data
glm_pred <- predict(newdata = test_data, glm_model, type = "response")
glm_pred <- ifelse(glm_pred > 0.5, 1, 0)
# Convert predicted data to factor with same levels as test data
glm_pred_factor <- factor(glm_pred, levels = levels(test_data$DEFAULT)) # Calculate confusion matrix
conf_mat_glm <- confusionMatrix(test_data$DEFAULT, glm_pred_factor)
```

#### Support Vector Machine (SVM)
Training the model:
```
svm_model <- svm(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + age_bins +
             PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
             PAY_AMT1+ PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6 +
             BILL_AMT1  + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 , data = train_data)
```
Calculating the confusion matrix: 
```
# Make predictions on test data
svm_pred <- predict(newdata = test_data, svm_model) # Calculate confusion matrix
conf_mat_svm <- confusionMatrix(test_data$DEFAULT, svm_pred)
```
#### Neural Network
Training the model:
```
nn_model <- nnet(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + age_bins +
PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
PAY_AMT1+ PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6 +
BILL_AMT1  + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6,
data = train_data, size = 5, decay = 5e-4, maxit = 500)
```
Calculating the confusion matrix: 
```
# Make predictions on test data
nn_pred <- predict(nn_model, newdata = test_data, type = "class") # Convert nn_pred to factor with the levels of test_data$DEFAULT
nn_pred <- factor(nn_pred, levels = levels(test_data$DEFAULT)) # Calculate confusion matrix
conf_mat_nn <- table(test_data$DEFAULT, nn_pred)
```

#### Random Forest
Training the model:
```
# Train random forest model
rf_model <- randomForest(DEFAULT ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + age_bins +
             PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
             PAY_AMT1+ PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6 +
             BILL_AMT1  + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6,data = train_data, ntree = 500, importance = TRUE)
```
Calculating the confusion matrix: 
```
# Make predictions on test data
rf_pred <- predict(rf_model, newdata = test_data) # Calculate confusion matrix
conf_mat_rf <- confusionMatrix(test_data$DEFAULT, rf_pred)
```
Plotting ROC curve:
```
# Create ROC curve
roc_rf <- roc(test_data$DEFAULT, predict(rf_model, newdata = test_data, type = "prob")[,2])
```
<img width="841" alt="Screenshot 2023-06-29 at 5 15 33 PM" src="https://github.com/thisissteff/credit_risk_analysis/assets/138084370/5d785878-864e-4358-85e6-908d7aac7df5">


## Selecting the Best Model
We tabulated the Accuracy, Precision, Sensitivity, Specificity, Average Class Accuracy as well as Harmonic Mean of each model. However, as mentioned above, the dataset is significantly imbalanced with more customers with non-default payments as compared to customers with default payments. Hence, classification accuracy can mask poor performance as the performance of the non-default level overwhelms the performance of the default level. Therefore, in this case, the Accuracy rate may not be a good metric to help us to identify the best model as it does not tell us the underlying distribution of the dataset. We use average class accuracy and the harmonic mean is computed as a secondary comparison. Harmonic mean helps to tackle biases by data imbalances by being sensitive to values that are lower than the average.

<img width="855" alt="Screenshot 2023-06-29 at 5 12 59 PM" src="https://github.com/thisissteff/credit_risk_analysis/assets/138084370/8e78ceba-28d3-40e7-8195-97e18e9204b0">

## Conclusion
Random Forest performs the best out of the other 3 models with the highest average class accuracy and harmonic mean and thus, based on our analysis we can conclude that it is the best model to predict the variable `DEFAULT`. We used k-fold cross-validation, where `K = 10` and we observed an accuracy of 0.814883 or 81.5%, which is higher than the initial accuracy of 0.8122149. Additionally, ROC index has a value of 0.7641886 which is greater than 0.7. Thus we can safely conclude that the selected model, Random Forest, is a strong model.


## Areas for Improvement
We could have consider other models such as **Naive Bayes (NB)** since it is also robust to irrelevant features, allowing it to provide good performance even if some features are redundant. We could also implement **Tomek links** of **Principal Component Analysis (PCA)**.




