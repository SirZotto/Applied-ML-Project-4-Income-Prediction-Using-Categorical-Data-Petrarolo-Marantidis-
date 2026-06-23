## Project 4: *Income Prediction Using Categorical Data*

### 🧠 Motivation

Many real-world datasets contain categorical (non-numeric) variables. In linear models, how these features are encoded directly impacts model quality and interpretability. This project focuses on the **UCI Adult dataset**, which includes both numerical and categorical features describing individuals' demographics and occupations, with the goal of predicting income.

The challenge is to explore how different encoding choices — from sparse one-hot to compact or target-aware strategies — affect performance. Importantly, **the appropriate encoding depends on the variable’s meaning**: not all categoricals should be treated the same.

### 🎯 Objectives

- Identify which categorical features are low- or high-cardinality  
- Analyze which variables have a meaningful order (e.g., `education`)  
- Implement and compare:
  - **One-hot encoding**:
    
  $$
  \text{OneHot}_k(x) = \mathbb{1}\{x = c_k\}
  $$
  
  - **Ordinal encoding**:
    
  $$
  \text{Ordinal}(x = c_k) = \text{rank}(c_k)
  $$
    
  - **Frequency encoding**:
    
  $$
  \text{Freq}(x = c_k) = \frac{\text{count}(x = c_k)}{N}
  $$
    
  - **Target encoding**:
    
  $$
  \text{Target}(x = c_k) = \mathbb{E}[y \mid x = c_k]
  $$

- Implement **leakage control** for target encoding using cross-validation:
  - Encode each fold using target statistics from other folds (not including the target of the encoded point)

### 📊 Dataset

- **UCI Adult Income Dataset**  
  - Predict whether an individual earns >\$50K/year  
  - Features: `age`, `education`, `occupation`, `native-country`, `workclass`, etc.  
  - [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

Target variable:

$$
y =
\begin{cases}
1 & \text{if income } > \$50K \\
0 & \text{otherwise}
\end{cases}
$$

### 🛠️ Suggested Models

- Logistic regression (with or without regularization)
- Optional: linear SVM

### ✅ Minimum Requirements

- Apply and compare at least two different encoding strategies:
  - One-hot and at least one of: ordinal or target  
- Handle missing values appropriately (e.g., `workclass = '?'`)  
- Use cross-validation to train and evaluate models  
- Report: accuracy, precision, recall
- Discuss:
  - Why some features are better suited to non-one-hot encoding  
  - How encoding affects model interpretability (via coefficients)  
  - Tradeoffs between sparsity, performance, and leakage

### 🚀 Optional Extensions

- Use smoothed target encoding:
  
$$
\text{SmoothTarget}(c) = \frac{n_c \cdot \bar{y}_c + \alpha \cdot \bar{y}}{n_c + \alpha}
$$
  
- Apply Lasso regression to compare sparsity across encodings  
- Visualize encoded feature space with PCA or UMAP  
- Analyze subgroup performance (e.g., gender or race)

### 📚 References

- Pargent et al. (2022) – [*Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features*](https://arxiv.org/pdf/2104.00629)
- Kuhn & Johnson (2019) – [*Feature Engineering and Selection*](http://www.feat.engineering)


