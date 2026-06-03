# Data

This project uses the Bank Marketing dataset from the UCI Machine Learning Repository.

Dataset link: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

The notebook downloads the dataset automatically if `bank-full.csv` is not available locally.

The raw CSV file is not included in this repository to keep the project lightweight and reproducible.



# Bank Marketing Subscription Prediction using Decision Tree Classification

## Project Overview

This project builds an end-to-end **Decision Tree Classification** model to predict whether a bank customer will subscribe to a term deposit.

The dataset is the **Bank Marketing dataset** from the UCI Machine Learning Repository. It is based on direct marketing phone campaigns conducted by a Portuguese banking institution.

The main goal is to answer this question:

> Can we use customer, financial and campaign-related information to predict whether a customer will subscribe to a term deposit?

This is a **binary classification problem**.

| Target value | Meaning |
|---|---|
| `yes` | Customer subscribed to a term deposit |
| `no` | Customer did not subscribe |

---

## Dataset

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

The dataset used in this project contains:

| Item | Value |
|---|---:|
| Number of records | 45,211 |
| Number of columns | 17 |
| Input features | 16 |
| Target variable | `y` |

The features include customer demographics, financial information, current campaign details and previous campaign outcomes.

### Feature Groups

| Feature group | Columns |
|---|---|
| Customer information | `age`, `job`, `marital`, `education` |
| Financial information | `default`, `balance`, `housing`, `loan` |
| Current campaign information | `contact`, `day`, `month`, `duration`, `campaign` |
| Previous campaign information | `pdays`, `previous`, `poutcome` |
| Target variable | `y` |

### Important Modelling Note

The `duration` variable was removed before model training.

This is because call duration is only known after the customer call is completed. If this variable is used for prediction, it can cause **data leakage** and make the model unrealistically strong. In a real marketing campaign, the bank would need to predict likely subscribers before making the call.

---

## Project Workflow

The project follows a complete machine learning workflow:

1. Dataset loading
2. Initial data exploration
3. Exploratory data analysis
4. Subscription rate analysis
5. Data preprocessing
6. One-hot encoding for categorical variables
7. Train-test split
8. Baseline Decision Tree using Gini impurity
9. Baseline Decision Tree using Entropy and Information Gain
10. Hyperparameter tuning using GridSearchCV
11. Cost-complexity pruning experiment
12. Model comparison
13. Final model selection
14. Confusion matrix and model interpretation

---

## Decision Tree Concepts Covered

This project was designed to practise the core theory behind Decision Trees.

### 1. Gini Impurity

Gini impurity measures how mixed the classes are inside a node.

The formula is:

$$
Gini = 1 - \sum_{i=1}^{C} p_i^2
$$

Where:

- \(C\) is the number of classes
- \(p_i\) is the proportion of samples belonging to class \(i\)

For this project, there are two classes:

- `yes`
- `no`

A node with mostly one class has low Gini impurity. A node with a strong mix of both classes has higher Gini impurity.

The Decision Tree tries to choose splits that reduce impurity and create purer child nodes.

---

### 2. Entropy

Entropy measures the disorder or uncertainty in a node.

The formula is:

$$
Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

Where:

- \(p_i\) is the proportion of samples in class \(i\)
- \(\log_2\) is the base-2 logarithm

If a node contains only one class, entropy is 0.  
If a node contains a balanced mix of classes, entropy is higher.

In this project, an Entropy-based Decision Tree was trained to compare its performance against the Gini-based Decision Tree.

---

### 3. Information Gain

Information Gain measures how much uncertainty is reduced after a split.

The formula is:

$$
Information\ Gain = Entropy(parent) - \sum_{j=1}^{k} \frac{N_j}{N} Entropy(child_j)
$$

Where:

- \(N\) is the total number of records in the parent node
- \(N_j\) is the number of records in child node \(j\)
- \(k\) is the number of child nodes created by the split

A higher Information Gain means the split is more useful.

In simple terms:

> A good split creates child nodes that are more pure than the parent node.

---

### 4. Overfitting in Decision Trees

Decision Trees can easily overfit when they are allowed to grow too deep.

An overfitted tree performs extremely well on training data but poorly on unseen test data.

In this project, the baseline Gini and Entropy models both achieved a training accuracy of 1.00, which showed that the unrestricted trees were memorising the training data.

---

### 5. Hyperparameters Used

The following Decision Tree hyperparameters were tuned:

| Hyperparameter | Meaning |
|---|---|
| `criterion` | Splitting method: `gini` or `entropy` |
| `max_depth` | Maximum depth of the tree |
| `min_samples_split` | Minimum number of samples required to split a node |
| `min_samples_leaf` | Minimum number of samples required in a leaf node |
| `class_weight` | Helps handle class imbalance |
| `ccp_alpha` | Cost-complexity pruning parameter |

---

## Exploratory Data Analysis

### Target Variable Distribution

![Target Variable Distribution](images/target_distribution.png)

The target variable is imbalanced. Most customers did not subscribe to a term deposit, while only a smaller percentage subscribed.

This means accuracy alone can be misleading. For this reason, precision, recall, F1-score and ROC-AUC were also used to evaluate the model.

---

### Age Distribution

![Age Distribution](images/age_distribution.png)

The age distribution shows that most customers are middle-aged, with fewer customers in the very young and very old age groups.

Age may still be useful for the Decision Tree because subscription behaviour can vary across customer life stages.

---

### Subscription by Job Type

![Subscription by Job Type](images/subscription_by_job.png)

The subscription count differs across job categories. This suggests that customer occupation may be useful for predicting term deposit subscription.

A Decision Tree can use this type of categorical feature by applying one-hot encoding before model training.

---

### Subscription by Marital Status

![Subscription by Marital Status](images/subscription_by_marital_status.png)

The marital status plot shows how subscription outcomes vary across married, single and divorced customers.

This feature may help the model capture differences in financial behaviour between customer groups.

---

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

The correlation heatmap shows relationships between numerical variables.

Decision Trees do not require strong linear relationships between variables and the target. However, correlation analysis is still useful for understanding the dataset before modelling.

---

## Financial Obligation Analysis

During EDA, financial obligation variables showed a clear relationship with subscription behaviour.

### Subscription Rate by Housing Loan

![Subscription by Housing Loan](images/financial_housing_subscription.png)

| Housing loan | Did not subscribe | Subscribed |
|---|---:|---:|
| No | 83.30% | 16.70% |
| Yes | 92.30% | 7.70% |

Customers without a housing loan had a much higher subscription rate than customers with a housing loan.

---

### Subscription Rate by Personal Loan

![Subscription by Personal Loan](images/financial_loan_subscription.png)

| Personal loan | Did not subscribe | Subscribed |
|---|---:|---:|
| No | 87.34% | 12.66% |
| Yes | 93.32% | 6.68% |

Customers without a personal loan were more likely to subscribe than customers with an existing personal loan.

---

### Subscription Rate by Credit Default

![Subscription by Credit Default](images/financial_default_subscription.png)

| Credit default | Did not subscribe | Subscribed |
|---|---:|---:|
| No | 88.20% | 11.80% |
| Yes | 93.62% | 6.38% |

Customers with no default history had a higher subscription rate than customers with default history.

Overall, the EDA suggests that customers with fewer financial obligations may be more likely to subscribe to a term deposit.

---

## Data Preprocessing

The preprocessing steps included:

- Converting the target variable `y` into binary format:
  - `yes` = 1
  - `no` = 0
- Removing the `duration` column to avoid data leakage
- Separating input features and target variable
- Splitting the data into training and testing sets
- Applying one-hot encoding to categorical features
- Passing numerical features without scaling

Decision Trees do not require feature scaling because they split data using threshold-based rules.

---

## Model Development

Three main Decision Tree modelling stages were used:

### 1. Baseline Gini Decision Tree

The first baseline model used Gini impurity as the splitting criterion.

This model achieved perfect training accuracy, which showed clear overfitting.

### 2. Baseline Entropy Decision Tree

The second baseline model used Entropy and Information Gain.

This model also achieved perfect training accuracy, meaning it was also overfitting.

### 3. Tuned Decision Tree

The tuned model used GridSearchCV to control tree complexity using:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `criterion`
- `class_weight`

This reduced overfitting and improved ROC-AUC.

### 4. Final Pruned Decision Tree Experiment

A pruning experiment was also performed using `ccp_alpha`.

The pruned model achieved higher accuracy, but its ROC-AUC dropped. It also performed worse at identifying actual subscribers. Because of this, the tuned Decision Tree was selected as the final model.

---

## Model Results

| Model | Train Accuracy | Test Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Baseline Gini | 1.0000 | 0.8329 | 0.6127 |
| Baseline Entropy | 1.0000 | 0.8306 | 0.6081 |
| Tuned Decision Tree | 0.8398 | 0.8370 | 0.7558 |
| Final Pruned Decision Tree | 0.8928 | 0.8932 | 0.5873 |

### Model Comparison Interpretation

The baseline Gini and Entropy models both achieved training accuracy of 1.00. This means they memorised the training data and overfitted.

The tuned Decision Tree achieved a better balance between training and testing performance. Its ROC-AUC improved to 0.7558, which shows better separation between subscribers and non-subscribers.

The final pruned tree had the highest test accuracy, but its ROC-AUC dropped to 0.5873. This means the model became too conservative and was weaker at identifying actual subscribers.

Therefore, the **Tuned Decision Tree** was selected as the final model.

---

## Final Model Performance

The selected tuned Decision Tree achieved:

| Metric | Value |
|---|---:|
| Training accuracy | 0.8398 |
| Testing accuracy | 0.8370 |
| ROC-AUC | 0.7558 |
| Class 1 recall | 0.55 |
| Class 1 F1-score | 0.44 |

Class 1 represents customers who subscribed to a term deposit.

The tuned model was selected because it identified more actual subscribers than the baseline models and had the strongest ROC-AUC score.

---

## Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows how many customers were correctly and incorrectly classified by the tuned Decision Tree.

For this marketing problem, class 1 recall is important because the business goal is to identify customers who are likely to subscribe. Missing too many actual subscribers would reduce the usefulness of the model.

---

## Feature Importance

![Feature Importance](images/feature_importance.png)

Feature importance shows which variables contributed most to the Decision Tree splits.

These features can help explain which customer or campaign attributes were most useful for prediction.

---

## Decision Tree Visualisation

![Decision Tree Visualisation](images/decision_tree_visual.png)

The Decision Tree visualisation shows the first few levels of the final tree.

This is one advantage of Decision Trees: they are more interpretable than many black-box machine learning models. The model can be explained as a sequence of rules.

---

## Tools and Libraries Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

## Repository Structure

```text
bank-marketing-decision-tree/
│
├── data/
│   ├── bank-full.csv
│   └── README.md
│
├── images/
│   ├── target_distribution.png
│   ├── age_distribution.png
│   ├── subscription_by_job.png
│   ├── subscription_by_marital_status.png
│   ├── correlation_heatmap.png
│   ├── financial_housing_subscription.png
│   ├── financial_loan_subscription.png
│   ├── financial_default_subscription.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── decision_tree_visual.png
│
├── notebooks/
│   └── Bank_marketing_DT.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-decision-tree.git
cd bank-marketing-decision-tree
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook notebooks/Bank_marketing_DT.ipynb
```

Then run the notebook cells from top to bottom.

---

## Key Learnings

This project helped me understand:

- How Decision Trees split data using Gini impurity and Entropy
- How Information Gain is used to choose better splits
- Why unrestricted Decision Trees can overfit
- How hyperparameters control tree complexity
- Why accuracy is not enough for imbalanced classification problems
- How recall and ROC-AUC help evaluate subscriber prediction
- How model interpretability can support business decision-making

---

## Future Improvements

Possible next steps:

- Compare Decision Tree with Random Forest
- Try Gradient Boosting or XGBoost
- Use SMOTE or other imbalance-handling methods
- Tune probability threshold for better subscriber recall
- Deploy the model using Streamlit
- Add a dashboard showing customer subscription insights

---

## Final Conclusion

This project shows how a Decision Tree Classifier can be used for a real-world bank marketing classification problem.

The baseline Decision Trees overfitted the training data, while the tuned Decision Tree gave a better balance between generalisation and subscriber detection.

The final selected model was the **Tuned Decision Tree** because it achieved the highest ROC-AUC and improved the model's ability to identify customers who subscribed to a term deposit.
