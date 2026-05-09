# ==========================================
# MACHINE LEARNING LAB PROJECT
# Diabetes Prediction System
# ==========================================

# ========== 1. IMPORT LIBRARIES ==========
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========== 2. DATASET GENERATION ==========
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'Glucose': np.random.randint(70, 200, n),
    'BloodPressure': np.random.randint(60, 120, n),
    'BMI': np.random.uniform(18, 40, n),
    'Age': np.random.randint(20, 70, n),
    'Insulin': np.random.randint(15, 276, n)
})

# Creating target variable (Outcome)
data['Outcome'] = np.where(
    (data['Glucose'] > 130) &
    (data['BMI'] > 30) &
    (data['Age'] > 40),
    1, 0
)

# Save dataset
data.to_csv('diabetes_data.csv', index=False)
print("\nDataset Created Successfully!\n")

# ========== 3. DATA EXPLORATION ==========
print("First 5 Rows:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# ========== 4. DATA VISUALIZATION ==========
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(data, hue='Outcome')
plt.show()

plt.figure()
sns.boxplot(x='Outcome', y='Glucose', data=data)
plt.title("Glucose vs Outcome")
plt.show()

plt.figure()
sns.histplot(data['BMI'], kde=True)
plt.title("BMI Distribution")
plt.show()

# ========== 5. DATA SPLITTING ==========
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nData Split Completed!\n")

# ========== 6. MODEL TRAINING & TESTING ==========
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    results[name] = acc
    
    print("\n==============================")
    print(f"Model: {name}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

# ========== 7. BEST MODEL ==========
best_model = max(results, key=results.get)
print("\n===================================")
print("Best Performing Model:", best_model)
print("Best Accuracy:", results[best_model])

# ========== 8. FEATURE IMPORTANCE ==========
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure()
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance (Random Forest)")
plt.show()

# ========== 9. CLASS DISTRIBUTION ==========
print("\nOutcome Distribution:")
print(data['Outcome'].value_counts())

plt.figure()
sns.countplot(x='Outcome', data=data)
plt.title("Class Distribution")
plt.show()

# ========== 10. CONCLUSION ==========
print("\nConclusion:")
print("Machine learning models were successfully applied to predict diabetes.")
print("Different algorithms were compared, and the best model was selected based on accuracy.")
print("Feature importance shows which factors most influence predictions.")
