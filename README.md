# ✈️ Airline Satisfaction Prediction - AI Project

This project uses various machine learning and deep learning models to predict customer satisfaction in airline travel, based on structured passenger data. It includes advanced preprocessing, feature engineering, visualization, model training, evaluation, and a full-featured GUI for interaction.

## 📁 Files Included

* `Ai_project.py` – Main project script that performs data preprocessing, model training, evaluation, visualization, and GUI setup.
* `train.csv` / `test.csv` – Required datasets (not included in repo; see below).

## 🔍 Features

* Data cleaning, imputation, and label encoding
* Feature selection via correlation analysis
* Handling class imbalance using **SMOTE**
* Multiple model training:

  * Logistic Regression
  * Random Forest
  * Decision Tree
  * K-Nearest Neighbors
  * XGBoost
  * Deep Learning using Keras
* Polynomial feature generation for non-linear learning
* Interactive GUI built with `tkinter`:

  * Model selection
  * Real-time prediction input
  * Performance metric display
  * Data visualization (heatmap, distributions, SMOTE effects)

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

**Dependencies:**

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* keras
* tensorflow
* xgboost
* tkinter (comes with most Python installations)

## ▶️ How to Run

1. Place `train.csv` and `test.csv` in the same directory as the script.
2. Run the Python script:

```bash
python Ai_project.py
```

3. The GUI will launch automatically.

## 📊 Visualizations

The GUI supports several exploratory plots:

* Correlation Heatmap
* Feature Distributions
* Target Class Distribution
* SMOTE Before & After Comparison

## 🧠 Models and Evaluation

Each model is trained using pipelines that include preprocessing steps and SMOTE balancing. Evaluation metrics are displayed live in the GUI:

* Accuracy
* Precision
* Recall
* F1 Score

## 🧪 Deep Learning

A feedforward neural network is implemented using Keras and trained with binary cross-entropy loss. Features are scaled and transformed with polynomial interaction terms.

## 📌 Notes

* SVM is present but commented out due to performance or GUI constraints.
* The GUI enables full user interaction for inference on new data points.

