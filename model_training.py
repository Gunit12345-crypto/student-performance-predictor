import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

# Create average score
data['average_score'] = (
    data['math score'] +
    data['reading score'] +
    data['writing score']
) / 3

# PASS Criteria
# Overall percentage >= 40
# AND each subject >= 30

data['result'] = np.where(
    (data['average_score'] >= 40) &
    (data['math score'] >= 30) &
    (data['reading score'] >= 30) &
    (data['writing score'] >= 30),
    1,
    0
)

# Features
X = data[[
    'math score',
    'reading score',
    'writing score'
]]

# Target
y = data['result']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_acc)

# Select best model
if lr_acc >= knn_acc:

    best_model = lr_model

    print("Selected Model: Logistic Regression")

else:

    best_model = knn_model

    print("Selected Model: KNN")

# Save model
pickle.dump(best_model, open('model.pkl', 'wb'))

# Save scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler saved successfully!")