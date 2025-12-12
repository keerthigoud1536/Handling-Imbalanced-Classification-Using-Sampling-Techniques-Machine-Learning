# Handling-Imbalanced-Classification-Using-Sampling-Techniques-Machine-Learning
This project demonstrates how to handle imbalanced datasets using under sampling and oversampling techniques, followed by training machine learning models and evaluating them using F1-score and AUC-ROC score.  The goal is to compare how different sampling strategies improve classification performance on highly imbalanced data.

# 1.Introduction

The aim of this project is to handle an imbalanced dataset, where one class has significantly fewer samples than the other.
Imbalanced data often causes machine learning models to perform poorly on the minority class.
To solve this, under-sampling and over-sampling techniques are applied along with multiple ML models.

# 2. Dataset

The dataset used is df_train.
It contains features (independent variables) and a target variable (dependent variable).
The target is imbalanced (one class has fewer records).

# 3. Steps Performed
**Step 1 – Read the Data**
Loaded dataset using pandas.
Checked:
Shape
Info (datatypes)
Null values
Class distribution

**Step 2 – EDA (Without Visualizations)**
Since no visualization was done, EDA included:
Summary statistics using describe().
Count of missing values.
Class imbalance check using:
df_train[target].value_counts()
Identified which features are numerical and categorical.
Checked correlations using df_train.corr() (no heatmap drawn).

**Step 3 – Install and Import imbalanced-learn**

Used for handling imbalance:
ClusterCentroids (under-sampling)
RandomOverSampler
SMOTE

**Step 4 – Apply Random Under-Sampling (Cluster Centroids)**

Reduced the majority class by creating cluster centroids.
Prevents losing information.
Balanced the dataset.
Used for the first ML model experiment.

**Step 5 – Train ML Models (Under-sampled Dataset)**

Models trained:
Logistic Regression
Decision Tree Classifier

#Evaluated using:
F1-score
AUC Score

**Classification Report**

**Step 6 – Evaluate Results**

Compared minority class performance before & after balancing.
Under-sampling usually results in:
Higher recall
Better F1-score
Slight drop in overall accuracy
This is expected because the goal is to improve minority class performance.

**Step 7 – Apply Random Over-Sampling & SMOTE**

Performed after under-sampling experiment:
RandomOverSampler → duplicates minority samples.
SMOTE → creates synthetic samples.
Balanced the dataset with more data.

**Step 8 – Apply ML Models Again (Over-sampled Dataset)**

Trained the same models again:
Logistic Regression
Decision Tree

Evaluated with same metrics:
F1-score,AUC


# 5.Technologies Used 

| Category      | Tools                          |
| ------------- | ------------------------------ |
| Programming   | Python                         |
| ML Libraries  | Scikit-learn, Imbalanced-learn |
| Data          | Pandas, NumPy                  |
| Visualization | Matplotlib, Seaborn            |

# 6.**Model Performance comparision table**


| Sampling Method                       | Model               | F1 Score    | AUC Score   |
| ------------------------------------- | ------------------- | ----------- | ----------- |
| **UnderSampling (Cluster Centroids)** | Logistic Regression | **0.07034** | **0.52522** |
| **UnderSampling (Cluster Centroids)** | Decision Tree       | **0.07033** | **0.50000** |
| **OverSampling (SMOTE)**              | Logistic Regression | **0.07486** | **0.54110** |
| **OverSampling (SMOTE)**              | Decision Tree       | **0.05898** | **0.50926** |

# **Final Conclusion**

-SMOTE + Logistic Regression / Decision Tree usually gives the best balanced performance.

-Cluster Centroids is useful but SMOTE performed better overall.

