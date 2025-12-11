# Handling-Imbalanced-Classification-Using-Sampling-Techniques-Machine-Learning
This project demonstrates how to handle imbalanced datasets using under sampling and oversampling techniques, followed by training machine learning models and evaluating them using F1-score and AUC-ROC score.  The goal is to compare how different sampling strategies improve classification performance on highly imbalanced data.

# 1.INTRODUCTION

The aim of this project is to handle an imbalanced dataset, where one class has significantly fewer samples than the other.
Imbalanced data often causes machine learning models to perform poorly on the minority class.
To solve this, under-sampling and over-sampling techniques are applied along with multiple ML models.

# 2. DATASET

The dataset used is df_train.
It contains features (independent variables) and a target variable (dependent variable).
The target is imbalanced (one class has fewer records).

# 3. STEPS PERFORMED
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
# ⭐SAMPLING TECHNIQUE COMPARISION

| Sampling Technique                            | Advantages                                                                           | Disadvantages                                                                   | Model Performance Impact                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Under-Sampling (Cluster Centroids)**        | - Faster training time  <br> - Good class balance                                    | - Reduces dataset size  <br> - May remove important patterns                    | - Moderate improvement in F1-score  <br> - Sometimes lower AUC due to data loss                 |
| **Over-Sampling (Random Oversampling - ROS)** | - Simple and fast <br> - Increases minority samples                                  | - Can overfit due to duplicate samples                                          | - Better F1-score than under-sampling <br> - AUC improves                                       |
| **Over-Sampling (SMOTE)**                     | - Creates synthetic, realistic samples <br> - Improves minority class representation | - Slightly slower than ROS <br> - Can create noisy samples if data is not clean | - **Best F1-score** <br> - **Best ROC-AUC performance** <br> - Most stable model generalization |


**Final Conclusion**

-SMOTE + Logistic Regression / Decision Tree usually gives the best balanced performance.
-Cluster Centroids is useful but SMOTE performed better overall.

# ⭐ TECHNOLOGIES USED 

| Category      | Tools                          |
| ------------- | ------------------------------ |
| Programming   | Python                         |
| ML Libraries  | Scikit-learn, Imbalanced-learn |
| Data          | Pandas, NumPy                  |
| Visualization | Matplotlib, Seaborn            |
