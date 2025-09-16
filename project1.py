import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.shape)
print(df.info())

df['Name'] = df['Name'].astype(str)
df['Name'] = df['Name'].astype('category')
df['Name'] = df['Name'].cat.codes 

df['Sex'] = df['Sex'].astype(str)
df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes 

df['Ticket'] = df['Ticket'].astype(str)
df['Ticket'] = df['Ticket'].astype('category')
df['Ticket'] = df['Ticket'].cat.codes 

df['Cabin'] = df['Cabin'].astype(str)
df['Cabin'] = df['Cabin'].astype('category')
df['Cabin'] = df['Cabin'].cat.codes 

df['Embarked'] = df['Embarked'].astype(str)
df['Embarked'] = df['Embarked'].astype('category')
df['Embarked'] = df['Embarked'].cat.codes 


print(df.describe())

dups=df.duplicated()
print("the number of duplicate rows= %d" %(dups.sum()))
df[dups]

df.boxplot()
plt.show()

def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR= Q3-Q1
    lower_range= Q1-(1.5*IQR)
    upper_range= Q3+(1.5*IQR)
    return lower_range,upper_range

lrage,urage=remove_outlier(df["Age"])
df["Age"]=np.where(df["Age"]>urage,urage,df["Age"])
df["Age"]=np.where(df["Age"]<lrage,lrage,df["Age"])

lrSibSp,urSibSp=remove_outlier(df["SibSp"])
df["SibSp"]=np.where(df["SibSp"]>urSibSp,urSibSp,df["Age"])
df["SibSp"]=np.where(df["SibSp"]<lrSibSp,lrSibSp,df["SibSp"])

lrParch,urParch=remove_outlier(df["Parch"])
df["Parch"]=np.where(df["Parch"]>urParch,urParch,df["Parch"])
df["Parch"]=np.where(df["Parch"]<lrParch,lrParch,df["Parch"])

lrFare,urFare=remove_outlier(df["Fare"])
df["Fare"]=np.where(df["Fare"]>urFare,urFare,df["Fare"])
df["Fare"]=np.where(df["Fare"]<lrFare,lrFare,df["Fare"])

lrCabin,urCabin=remove_outlier(df["Cabin"])
df["Cabin"]=np.where(df["Cabin"]>urCabin,urCabin,df["Cabin"])
df["Cabin"]=np.where(df["Cabin"]<lrCabin,lrCabin,df["Cabin"])

df.boxplot()
plt.show()

print(df.isnull().sum())

median1=df["Age"].median()
median2=df["SibSp"].median()

df["Age"].replace(np.nan,median1,inplace=True)
df["SibSp"].replace(np.nan,median2,inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Cabin"].fillna(df["Cabin"].mode()[0], inplace=True)

print(df.isnull().sum())

sns.displot(df.PassengerId,bins=20)
plt.show()

sns.displot(df.Survived,bins=20)
plt.show()

sns.displot(df.Pclass,bins=20)
plt.show()

sns.displot(df.Name,bins=20)
plt.show()

sns.displot(df.Sex,bins=20)
plt.show()

sns.displot(df.Age,bins=20)
plt.show()

sns.displot(df.SibSp,bins=20)
plt.show()

sns.displot(df.Parch,bins=20)
plt.show()

sns.displot(df.Ticket,bins=20)
plt.show()

sns.displot(df.Fare,bins=20)
plt.show()

sns.displot(df.Cabin,bins=20)
plt.show()

sns.displot(df.Embarked,bins=20)
plt.show()

sns.pairplot(df)
plt.show()

print(df.corr())

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,fmt='0.2f',cmap='Blues')
plt.show()
 
from sklearn.preprocessing import StandardScaler
std_scale=StandardScaler()
std_scale
StandardScaler(copy=True,with_mean=True,with_std=True)
df["PassengerId"]=std_scale.fit_transform(df[["PassengerId"]])
df["Survived"]=std_scale.fit_transform(df[["Survived"]])
df["Pclass"]=std_scale.fit_transform(df[["Pclass"]])
df["Name"]=std_scale.fit_transform(df[["Name"]])
df["Sex"]=std_scale.fit_transform(df[["Sex"]])
df["Age"]=std_scale.fit_transform(df[["Age"]])
df["SibSp"]=std_scale.fit_transform(df[["SibSp"]])
df["Parch"]=std_scale.fit_transform(df[["Parch"]])
df["Ticket"]=std_scale.fit_transform(df[["Ticket"]])
df["Fare"]=std_scale.fit_transform(df[["Fare"]])
df["Cabin"]=std_scale.fit_transform(df[["Cabin"]])
df["Embarked"]=std_scale.fit_transform(df[["Embarked"]])

print(df.head())
