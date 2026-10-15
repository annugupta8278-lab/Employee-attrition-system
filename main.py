
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df)
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500


df.Gender.value_counts()

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df.head())

print(df.head())


df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'], dtype=int)

df = df.map(lambda x: 1 if x is True else 0 if x is False else x)
print(df.head())

df = df.drop(['EmployeeNumber'], axis=1)

df.hist(figsize=(20, 15))
plt.tight_layout()
plt.show()

df = df.drop(['EmployeeCount', 'Over18' ,'StandardHours'], axis=1)
print(df.head())

#model training!!!!!!

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = df.drop('Attrition', axis=1), df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_jobs=-1)

model.fit(X_train, y_train)
print("Model Accuracy:", model.score(X_test, y_test))


sorted_importances = dict(sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda x: x[1], reverse=True))


plt.figure(figsize=(20, 8))
plt.bar(sorted_importances.keys(), sorted_importances.values())
plt.xticks(rotation=45, ha='right')

plt.show()

import joblib

# Save the trained model
joblib.dump(model, 'employee_model.pkl')
print("Model saved successfully!")