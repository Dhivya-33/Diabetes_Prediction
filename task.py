# ===============================
# Diabetes Prediction with SVM
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv("diabetes.csv")  # Ensure the CSV is in your working directory

# 3. Exploratory Data Analysis (EDA)
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Check target balance
sns.countplot(x='Outcome', data=df)
plt.title("Distribution of Diabetes Outcome")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 4. Data Preprocessing
# Replace zeros with NaN for specific columns
cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train SVM Model
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)

# 7. Predictions and Evaluation
y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate best model
best_model = grid.best_estimator_
y_best_pred = best_model.predict(X_test)

print("\nAccuracy after GridSearchCV:", accuracy_score(y_test, y_best_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_best_pred))
print("\nClassification Report:\n", classification_report(y_test, y_best_pred))

# 9. Visualize Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_best_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Best SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
