
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("SalaryData.csv")

print(data.head())
print("Dataset Shape:", data.shape)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())

plt.figure(figsize=(6,4))
plt.hist(data["Salary"], bins=10, color="skyblue", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.show()


plt.figure(figsize=(6,4))
plt.scatter(data["Years of Experience"], data["Salary"], color="red")
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


plt.figure(figsize=(6,4))
data["Education Level"].value_counts().plot(kind="bar", color="green")
plt.title("Education Level Count")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,4))
data.groupby("Job Title")["Salary"].mean().plot(kind="bar", color="orange")
plt.title("Average Salary by Job Title")
plt.xlabel("Job Title")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.show()


data.fillna(data.mean(numeric_only=True), inplace=True) #preprocessing

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["Education Level"] = le_education.fit_transform(data["Education Level"])
data["Job Title"] = le_job.fit_transform(data["Job Title"])

X = data.drop("Salary", axis=1)
y = data["Salary"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

prediction = model.predict(X_test)

print("R2 score:", r2_score(Y_test, prediction))
print("Mean absolute error:", mean_absolute_error(Y_test, prediction))
print("Mean squared error:", mean_squared_error(Y_test, prediction))

result = pd.DataFrame({
    "Actual Salary": Y_test,
    "Predicted Salary": prediction
})

print(result.head(10))

print("Model Saved Successfully")