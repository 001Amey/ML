import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())

print(df.tail())

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

# corr = df.corr()
# print(corr.Survived.sort_values(ascending=False).head(10))

print(df["Embarked"].unique())

df["Embarked"] = df["Embarked"].replace("S",0)
df["Embarked"] = df["Embarked"].replace("C",1)
df["Embarked"] = df["Embarked"].replace("Q",2)
print(df.head())

print(df["Age"].fillna(df["Age"].median(),inplace=True))

print(df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True))

df.drop("Cabin",axis=1,inplace=True)

print(df.duplicated().sum())

print(df.head())

sns.boxplot(x=df["Age"])
plt.show()

sns.boxplot(x=df["Fare"])
plt.show()

plt.figure(figsize=(14,8))
df['Sex'].value_counts().plot(kind="pie",autopct="%.2f")
plt.show()


le=LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print(df.head())

sns.histplot(df["Age"],kde=True)
plt.show()

sns.countplot(x="Survived",data=df)
plt.show()

sns.countplot(x="Sex",hue="Survived",data=df)
plt.show()

sns.scatterplot(x="Age",y="Fare",data=df)
plt.show()

plt.figure(figsize=(10,8))
numeric_df=df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
plt.show()

sns.pairplot(df)
plt.show()

scaler = StandardScaler()
df[["Age","Fare"]] = scaler.fit_transform(df[["Age","Fare"]])
print(df.head())

X=df.drop("Survived",axis=1)

y= df["Survived"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

Q1= df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3+1.5*IQR
df=df[(df["Fare"]>=lower)&(df["Fare"]<=upper)]

sns.boxplot(x=df["Fare"])
plt.show()


