# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC  # SVM commented as requested
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout

from xgboost import XGBClassifier

import tkinter as tk
from tkinter import ttk, messagebox

# --- Data preprocessing ---
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df.drop(full_df.columns[0], axis=1, inplace=True)
if 'id' in full_df.columns:
    full_df.drop('id', axis=1, inplace=True)

num_cols = full_df.select_dtypes(include=['number']).columns
cat_cols = full_df.select_dtypes(include=['object']).columns

full_df[num_cols] = SimpleImputer(strategy='mean').fit_transform(full_df[num_cols])
full_df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(full_df[cat_cols])

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])
    encoders[col] = le

corrs = full_df.corr()['satisfaction'].drop('satisfaction')
strong_corr = corrs[abs(corrs) > 0.3]
print("\n Strong Correlations (>|0.3|):")
print(strong_corr.sort_values(ascending=False))

top_corr_features = list(strong_corr.index)
X = full_df[top_corr_features]
y = full_df['satisfaction'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=len(test_df)/len(full_df), stratify=y, random_state=42
)

# --- Visualization Functions ---
def plot_heatmap():
    plt.figure(figsize=(10, 8))
    corr_features = list(strong_corr.index) + ['satisfaction']
    sns.heatmap(full_df[corr_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap (Strong Features Only)")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions():
    features_to_plot = full_df.drop('satisfaction', axis=1).columns
    for col in features_to_plot:
        plt.figure(figsize=(8, 4))
        sns.histplot(full_df[col], kde=True, bins=30)
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

def plot_target_distribution():
    plt.figure(figsize=(6, 4))
    sns.countplot(x=full_df['satisfaction'])
    plt.title("Target Variable Distribution: 'satisfaction'")
    plt.xlabel("Satisfaction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_smote_distribution():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_local = X_scaled[:len(train_df)]
    y_train_local = y[:len(train_df)]

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train_local)
    plt.title("Class Distribution Before SMOTE")
    plt.xlabel("Satisfaction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_local, y_train_local)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train_bal)
    plt.title("Class Distribution After SMOTE")
    plt.xlabel("Satisfaction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --- Model definitions with Pipeline ---
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest':        RandomForestClassifier(n_estimators=100),
    'Decision Tree':        DecisionTreeClassifier(),
    'KNN':                  KNeighborsClassifier(),
    # 'SVM':                  SVC(kernel='linear', probability=True),  # Commented out
    'XGBoost':              XGBClassifier(eval_metric='logloss')
}

models = {}
for name, estimator in base_models.items():
    models[name] = ImbPipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', estimator)
    ])
    models[name].fit(X_train, y_train)

# --- Deep Learning Model ---
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)

scaler = StandardScaler()
Xp_train = scaler.fit_transform(Xp_train)
Xp_test = scaler.transform(Xp_test)

dl_model = Sequential([
    Input(shape=(Xp_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(Xp_train, y_train, validation_data=(Xp_test, y_test), epochs=10, batch_size=32, verbose=1)
models['Deep Learning'] = dl_model

# --- Metrics Function ---
def get_metrics(model_name):
    if model_name == 'Deep Learning':
        preds = (models[model_name].predict(Xp_test) > 0.5).astype(int).flatten()
    else:
        preds = models[model_name].predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, average='weighted'),
        'Recall': recall_score(y_test, preds, average='weighted'),
        'F1 Score': f1_score(y_test, preds, average='weighted')
    }

# --- GUI ---
def run_full_input_gui(models, encoders, X_test):
    feats = X.columns.tolist()
    root = tk.Tk()
    root.title("Airline Satisfaction GUI")
    root.geometry("1000x1000")
    root.configure(bg='lightblue')

    tk.Label(root, text="Select Model:", bg='lightblue').grid(row=0, column=0, padx=10, pady=10)
    model_var = tk.StringVar(value=list(models.keys())[0])
    dropdown = ttk.Combobox(root, values=list(models.keys()), textvariable=model_var, state='readonly')
    dropdown.grid(row=0, column=1, pady=10)

    metrics_box = tk.Text(root, width=50, height=5)
    metrics_box.grid(row=1, column=0, columnspan=2, padx=10)

    def update_metrics(e=None):
        m = model_var.get()
        res = get_metrics(m)
        metrics_box.delete('1.0', tk.END)
        metrics_box.insert(tk.END, f"Metrics for {m}:\n")
        for k, v in res.items():
            metrics_box.insert(tk.END, f"{k}: {v:.4f}\n")

    dropdown.bind('<<ComboboxSelected>>', update_metrics)
    update_metrics()

    # Buttons for visualizations
    ttk.Button(root, text="Plot Heatmap", command=plot_heatmap).grid(row=2, column=2, padx=10, pady=5)
    ttk.Button(root, text="Feature Distributions", command=plot_feature_distributions).grid(row=3, column=2, padx=10, pady=5)
    ttk.Button(root, text="Target Distribution", command=plot_target_distribution).grid(row=4, column=2, padx=10, pady=5)
    ttk.Button(root, text="SMOTE Distribution", command=plot_smote_distribution).grid(row=5, column=2, padx=10, pady=5)

    inputs = {}
    start = 6
    for i, feat in enumerate(feats):
        tk.Label(root, text=feat, bg='lightblue').grid(row=start+i, column=0, sticky='e', padx=5)
        if feat in encoders:
            cb = ttk.Combobox(root, values=list(encoders[feat].classes_), state='readonly')
            cb.set(encoders[feat].classes_[0])
            cb.grid(row=start+i, column=1, padx=5)
            inputs[feat] = cb
        else:
            ent = ttk.Entry(root)
            ent.grid(row=start+i, column=1, padx=5)
            inputs[feat] = ent

    def predict():
        vals = [
            encoders[f].transform([inputs[f].get()])[0] if f in encoders else float(inputs[f].get())
            for f in feats
        ]
        df = pd.DataFrame([vals], columns=feats)
        pred_model = model_var.get()
        if pred_model == 'Deep Learning':
            xp = poly.transform(df)
            xp = scaler.transform(xp)
            pred = (models[pred_model].predict(xp) > 0.5).astype(int)[0][0]
        else:
            pred = models[pred_model].predict(df)[0]
        label = 'Satisfied' if pred == 1 else 'Dissatisfied'
        messagebox.showinfo("Result", f"{pred_model} predicts: {label}")
        update_metrics()

    ttk.Button(root, text="Predict", command=predict).grid(row=start+len(feats), column=0, columnspan=2, pady=10)
    root.mainloop()

run_full_input_gui(models, encoders, X_test)