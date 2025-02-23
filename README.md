OBESITY MULTICLASS PREDICTION

# Obesity Classification Model Analysis

## Introduction
Obesity is a growing public health concern in Rwanda, associated with numerous non-communicable diseases and increased mortality rates. This project aims to address the obesity crisis by developing machine learning models capable of early risk prediction. By using and manipulation of a dataset rich in demographic, lifestyle, and health indicators, we evaluate the performance of various classification models(XGBoost, SVM, RandomForest) and Neural Network to determine the most effective approach for obesity classification.

## Dataset Overview
This dataset contains 20,758 records and 17 features related to demographic, lifestyle, and health indicators. The target variable (NObeyesdad) classifies individuals into seven obesity categories:
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

  Below is a brief description of the variables

  id = Identifier for each individual in the dataset

  Gender = [Male or Female]

  Age = Individual's age in years

  Weight = Individual's weight

  Height = individual's Height in metres

  Health and Lifestyle Factors

  Family_history_with_overweight = Whether the family has history with overweight (Yes/No)

  FAvC = Frequent consumption of high calorie food. Indicates if the an individual frequently eats high-calorie food (Yes/No).

  FCVC = Frequency of consumption of vegetables, how often a person eats vegetables

  NCP = Number of main meals, number of main meals a person takes per day

  CAEC = (Consumption of Food Between Meals): How often the person snacks between meals (like "Sometimes", "Frequently", or "Always").

  SMOKE = Whether the person smokes (Yes/No).

  CH2O = (Daily Water Intake): How much water the person drinks daily

  SCC = (Caloric Consumption Monitoring): Whether the person monitors their caloric intake (Yes/No).

  FAF = (Physical Activity Frequency): How often the person exercises

  TUE = (Time Using Technology Devices): How much time the person spends using electronic devices daily

  CALC - (Alcohol Consumption): Frequency of alcohol consumption ("Never", "Sometimes", "Frequently").

  MTRANS (Mode of Transportation): How the person usually gets around (like "Walking", "Public Transport", "Car")

  Target Variable

  NObeyesdad: The classification of the individual’s weight condition — categories : "Normal Weight", "Overweight", "Obesity Type I", "Obesity Type II
  
Data Preprocessing

Binary encoding for categorical variables (Gender, Family History,SMOKE)

Ordinal encoding for ordered categories (CAEC, CALC)

One-hot encoding for transportation mode

Label encoding for target variable

Data split: 70% training, 15% validation, 15% test

Below is the process of how I handled categorical data to Numerical data

### Encoded Variables
- **Binary Categorical Columns:**
  - `Gender`: Male = 1, Female = 0
  - `family_history_with_overweight`: Yes = 1, No = 0
  - `FAVC` (Frequent consumption of high-caloric food): Yes = 1, No = 0
  - `SMOKE`: Yes = 1, No = 0
  - `SCC` (Calories consumption monitoring): Yes = 1, No = 0

- **Ordinal Categorical Columns:**
  - `CAEC` (Consumption of food between meals): Never = 0, Sometimes = 1, Frequently = 2, Always = 3
  - `CALC` (Consumption of alcohol): Never = 0, Sometimes = 1, Frequently = 2, Always = 3

- **One-Hot Encoded Columns:**
  - `MTRANS` (Transportation method): Converted to multiple binary columns

- **Target Variable Encoding:**
  - Categories mapped from 0 (Insufficient_Weight) to 6 (Obesity_Type_III)

## Model Performance Comparison

| Model            | Accuracy | Precision | Recall | F1 Score | Log Loss |
|------------------|----------|-----------|--------|----------|----------|
| **XGBoost**      | 0.91     | 0.91      | 0.91   | 0.91     | 0.2770   |
| **Random Forest**| 0.90     | 0.90      | 0.90   | 0.90     | 0.3785   |
| **Neural Network**| 0.88    | 0.88      | 0.88   | 0.88     | 0.3740   |
| **SVM**          | 0.86     | 0.86      | 0.86   | 0.86     | 0.3998   |

## Detailed Model Results

### XGBoost Model Results
| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.90  |
| Precision    | 0.90  |
| Recall       | 0.90  |
| F1 Score     | 0.90  |
| Log Loss     | 0.3199|

**Validation Classification Report:**
| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.91   | 0.93     | 376     |
| 1     | 0.86      | 0.90   | 0.88     | 441     |
| 2     | 0.79      | 0.78   | 0.79     | 388     |
| 3     | 0.79      | 0.80   | 0.80     | 377     |
| 4     | 0.88      | 0.87   | 0.87     | 422     |
| 5     | 0.96      | 0.97   | 0.97     | 509     |
| 6     | 0.99      | 1.00   | 0.99     | 601     |

### SVM Results
| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.86  |
| Precision    | 0.86  |
| Recall       | 0.86  |
| F1 Score     | 0.86  |
| Log Loss     | 0.3998|

**Validation Classification Report:**
| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.91      | 0.88   | 0.89     | 376     |
| 1     | 0.79      | 0.80   | 0.80     | 441     |
| 2     | 0.67      | 0.72   | 0.69     | 388     |
| 3     | 0.70      | 0.72   | 0.71     | 377     |
| 4     | 0.88      | 0.80   | 0.84     | 422     |
| 5     | 0.95      | 0.98   | 0.96     | 509     |
| 6     | 1.00      | 1.00   | 1.00     | 601     |

### Random Forest Results
| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.89  |
| Precision    | 0.89  |
| Recall       | 0.89  |
| F1 Score     | 0.89  |
| Log Loss     | 0.3968|

### Neural Network Results
| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.88  |
| Precision    | 0.88  |
| Recall       | 0.88  |
| F1 Score     | 0.88  |
| Loss         | 0.3740|

## Overall Summary
XGBoost achieved the highest accuracy and best overall performance with 91% accuracy and the lowest log loss (0.2770). Random Forest closely followed with 90% accuracy. Neural networks performed well with 88% accuracy after careful hyperparameter tuning, and SVM showed competitive performance but lagged slightly behind at 86%. XGBoost stands out as the most suitable model for deployment given its efficiency and superior performance in handling non-linear relationships.




## Neural Network Optimization Experiments

**Training Configurations and Results**

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Learning Rate | Dropout | Accuracy | F1 Score | Recall | Precision |
|----------|-----------|-------------|--------|----------------|---------------|---------|----------|----------|--------|-----------|
| 1        | None      | None        | 50     | No             | N/A           | No      | 0.10     | 0.17     | 0.20   | 0.23      |
| 2        | Adam      | None        | 50     | No             | 0.01          | 0.20    | 0.85     | 0.85     | 0.85   | 0.85      |
| 3        | Adam      | L2          | 500    | Yes            | 0.01          | 0.30    | 0.82     | 0.83     | 0.85   | 0.85      |
| 4        | RMSProp   | None        | 50     | No             | 0.001         | 0.30    | 0.875    | 0.880    | 0.880  | 0.885     |
| 5        | Adam      | None        | 50     | Yes            | 0.0001        | 0.30    | 0.885    | 0.878    | 0.880  | 0.880     |


 **Analysis of Optimization Techniques**

 #Instance 1

No optimization techniques

10% accuracy, near-random performance

The poor performance demonstrates the necessity and need of optimization in handling the relationships between health factors and obesity classification. Without proper weight updates, the model failed to learn meaningful patterns, there by guessing terribly

#Instances 2, 4, 5

**Adam vs RMSProp**

  RMSProp achieved higher accuracy (87.5%) compared to Adam with same epochs (85%)

  Adam with tuned learning rate (Instance 5) eventually performed better than RMSProp, this is due to Adam's  works better with proper hyperparameter tuning

**Learning Rate Optimization (Instances 2, 4, 5)**

Progressive reduction from 0.01 → 0.001 → 0.0001 showed consistent improvement
Best performance (88.45%) achieved with 0.0001
Analysis: Smaller learning rates allowed:

More precise weight updates for capturing subtle feature relationships
Better convergence in the complex obesity classification landscape
Reduced risk of overshooting optimal weights

**Regularization Impact (Instance 3)**

L2 regularization slightly decreased accuracy but improved generalization
Accuracy dropped from 85% to 82%
Maintained consistent recall and precision (0.85)

**Model Comparison and Hyperparameters **

XGBoost emerged as the best-performing model with the highest accuracy (91%) and lowest log loss (0.2770). Random Forest followed closely with 90% accuracy. While the Neural Network performed well with 88% accuracy, XGBoost’s efficiency and ability to manage non-linear relationships make it preferable.

XGBoost Hyperparameters:

n_estimators: Number of boosting rounds

max_depth: Maximum depth of each tree

learning_rate: Step size shrinkage to prevent overfitting

subsample: Proportion of data used for each boosting round

colsample_bytree: Fraction of features used per tree

gamma: Minimum loss reduction required to split a node

For XGBoost, the following hyperparameters were used for grid search:

param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0]
}

Neural Network required more tuning but did not outperform XGBoost’s default hyperparameter optimization.



