import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

df = pd.read_csv(r"Data/raw/Telco-Customer-Churn.csv")

print("\nDataset Loaded Successfully")

print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe(include="all"))

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.show()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df.drop_duplicates(inplace=True)

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

num_cols = df.select_dtypes(include=["int64","float64"]).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    if col != "Churn":
        plt.figure(figsize=(7,4))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        plt.title(col)
        plt.tight_layout()
        plt.show()

temp = df.copy()

temp = pd.get_dummies(temp, drop_first=True)

plt.figure(figsize=(18,12))
sns.heatmap(temp.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder

print("\nEncoding Completed")

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape :", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("\nPreprocessing Completed")
print("=" * 60)
print("=" * 60)


print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model_scores = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    model_scores[name] = acc

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)
    print("Accuracy :", round(acc,4))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(name)
    plt.show()

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

result = pd.DataFrame({
    "Model": model_scores.keys(),
    "Accuracy": model_scores.values()
})

result = result.sort_values(by="Accuracy", ascending=False)

print(result.to_string(index=False))

plt.figure(figsize=(8,5))
sns.barplot(data=result, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.show()

best_model_name = result.iloc[0]["Model"]

print("\nBest Model :", best_model_name)

if best_model_name == "Logistic Regression":
    best_model = LogisticRegression(max_iter=1000)

elif best_model_name == "KNN":
    best_model = KNeighborsClassifier()

elif best_model_name == "Decision Tree":
    best_model = DecisionTreeClassifier(random_state=42)

else:
    best_model = RandomForestClassifier(random_state=42)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[5,10,20,None],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}

if best_model_name == "Random Forest":

    grid = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    print("\nBest Parameters")
    print(grid.best_params_)

    print("\nBest CV Score")
    print(grid.best_score_)

else:

    best_model.fit(X_train,y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

print("Accuracy :",accuracy)

print("\nClassification Report")
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
    )
plt.title("Final Model")
plt.show()

print("\nPART 2 COMPLETED")


import os

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs("Models", exist_ok=True)

joblib.dump(best_model, "Models/model.pkl")
joblib.dump(scaler, "Models/scaler.pkl")
joblib.dump(label_encoders, "Models/label_encoders.pkl")

print("Model Saved Successfully")
print("Scaler Saved Successfully")
print("Label Encoders Saved Successfully")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if best_model_name == "Random Forest":

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=importance.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Important Features")
    plt.show()

print("\n" + "=" * 60)
print("SINGLE CUSTOMER PREDICTION")
print("=" * 60)

sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)

probability = best_model.predict_proba(sample_scaled)



if prediction[0] == 1:
    print("Prediction : Customer Will Churn")
else:
    print("Prediction : Customer Will Not Churn")

print("Probability :", round(np.max(probability) * 100, 2), "%")

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

for model, score in model_scores.items():
    print(f"{model:<25} : {score:.4f}")

print("\nBest Model :", best_model_name)
print("Final Accuracy :", round(accuracy,4))

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

