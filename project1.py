import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === LOAD DATA ===
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.shape)
print(df.info())

# === CATEGORY ENCODING (CLEAN) ===
cat_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].astype('category').cat.codes

# === BASIC STATS ===
print(df.describe())

# === DUPLICATES ===
dups = df.duplicated()
print("the number of duplicate rows= %d" % (dups.sum()))
df[dups]

# === BOX PLOT BEFORE OUTLIER HANDLING ===
df.boxplot()
plt.show()

# === OUTLIER REMOVAL ===
def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

lrage, urage = remove_outlier(df["Age"])
df["Age"] = np.where(df["Age"] > urage, urage, df["Age"])
df["Age"] = np.where(df["Age"] < lrage, lrage, df["Age"])

lrSibSp, urSibSp = remove_outlier(df["SibSp"])
df["SibSp"] = np.where(df["SibSp"] > urSibSp, urSibSp, df["SibSp"])
df["SibSp"] = np.where(df["SibSp"] < lrSibSp, lrSibSp, df["SibSp"])

lrParch, urParch = remove_outlier(df["Parch"])
df["Parch"] = np.where(df["Parch"] > urParch, urParch, df["Parch"])
df["Parch"] = np.where(df["Parch"] < lrParch, lrParch, df["Parch"])

lrFare, urFare = remove_outlier(df["Fare"])
df["Fare"] = np.where(df["Fare"] > urFare, urFare, df["Fare"])
df["Fare"] = np.where(df["Fare"] < lrFare, lrFare, df["Fare"])

lrCabin, urCabin = remove_outlier(df["Cabin"])
df["Cabin"] = np.where(df["Cabin"] > urCabin, urCabin, df["Cabin"])
df["Cabin"] = np.where(df["Cabin"] < lrCabin, lrCabin, df["Cabin"])

# === BOX PLOT AFTER OUTLIER HANDLING ===
df.boxplot()
plt.show()

# === MISSING VALUES HANDLING ===
print(df.isnull().sum())

# Median Imputation
median_age = df["Age"].median()
median_sibsp = df["SibSp"].median()

df["Age"].replace(np.nan, median_age, inplace=True)
df["SibSp"].replace(np.nan, median_sibsp, inplace=True)

# Mode Imputation (for categorical-like features)
mode_embarked = df["Embarked"].mode()[0]
mode_cabin = df["Cabin"].mode()[0]

df["Embarked"].replace(np.nan, mode_embarked, inplace=True)
df["Cabin"].replace(np.nan, mode_cabin, inplace=True)

print(df.isnull().sum())

# === HISTOGRAMS/DISTPLOTS ===
sns.displot(df.PassengerId, bins=20); plt.show()
sns.displot(df.Survived, bins=20); plt.show()
sns.displot(df.Pclass, bins=20); plt.show()
sns.displot(df.Name, bins=20); plt.show()
sns.displot(df.Sex, bins=20); plt.show()
sns.displot(df.Age, bins=20); plt.show()
sns.displot(df.SibSp, bins=20); plt.show()
sns.displot(df.Parch, bins=20); plt.show()
sns.displot(df.Ticket, bins=20); plt.show()
sns.displot(df.Fare, bins=20); plt.show()
sns.displot(df.Cabin, bins=20); plt.show()
sns.displot(df.Embarked, bins=20); plt.show()

# === PAIRPLOT ===
sns.pairplot(df)
plt.show()

# === CORRELATION  HEATMAP ===
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, fmt='0.2f', cmap='Blues')
plt.show()

# === SCALING ===
std_scale = StandardScaler()
cols_to_scale = df.columns  # scale all numeric columns
df[cols_to_scale] = std_scale.fit_transform(df[cols_to_scale])

# === FINAL PREVIEW ===
print(df.head())
