import pandas as pd
df = pd.read_csv(r"C:\Users\23196\Downloads\train.csv")
df.head()

!pip install seaborn

#1.Exploratory Data Analysis/Анализ разведочных данных
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="pastel")

df = pd.read_csv(r"C:\Users\23196\Downloads\train.csv")
df.head()

#2.Основная информация
#Количество строк, количество столбцов
df.shape
#Тип данных, ситуация с пропущенным значением
df.info()

#3.Подсчитайте количество пропущенных значений
missing = df.isnull().sum().sort_values(ascending=False)
missing.head(20)
#коэффициент пропущенных значений
missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_percent.head(20)

#4.Статистика целевой переменной SalePrice
#Описательная статистика
df['SalePrice'].describe()
#Карта распространения + KDE
plt.figure(figsize=(10,6))
sns.histplot(df['SalePrice'], kde=True, color='steelblue')
plt.title("Распределение стоимости жилья (SalePrice)")
plt.show()

#5. Корреляционный анализ числовых переменных
#Матрица корреляции
corr = df.corr(numeric_only=True)
corr['SalePrice'].sort_values(ascending=False).head(20)
#Тепловая карта
plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.show()

#6. Визуализация переменных, тесно коррелирующих с ценами на жилье
#Общее качество против цены продажи (уровень качества)
plt.figure(figsize=(8,6))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title("Влияние качества дома (OverallQual) на стоимость")
plt.show()
#GrLivArea против SalePrice (жилая площадь над землей)
plt.figure(figsize=(8,6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title("GrLivArea vs SalePrice")
plt.show()
#GarageCars vs SalePrice (количество парковочных мест)
plt.figure(figsize=(8,6))
sns.boxplot(x='GarageCars', y='SalePrice', data=df)
plt.title("Количество машин в гараже и стоимость дома")
plt.show()

#7. Влияние категориальных переменных на цены (на примере района)
plt.figure(figsize=(14,6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
plt.xticks(rotation=60)
plt.title("Район (Neighborhood) и стоимость")
plt.show()

#8. Изучите взаимосвязи между переменными площади.
numeric_cols = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']

sns.pairplot(df[numeric_cols + ['SalePrice']])
plt.show()
