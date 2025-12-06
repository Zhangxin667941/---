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

#Чистка данных
#1. Проверка пропусков
missing = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

pd.DataFrame({"Missing": missing, "%": missing_percent}).head(20)
#Определяем переменные с наибольшим количеством пропусков и вычисляем процент отсутствующих значений.

#2. Обработка категориальных пропусков
none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]

for col in none_cols:
    df[col] = df[col].fillna("None")
#Во многих категориальных колонках пропуск означает «отсутствует». Поэтому заменяем NaN на значение "None".

#3. Обработка числовых пропусков
zero_cols = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea'
]

for col in zero_cols:
    df[col] = df[col].fillna(0)
#Если отсутствует гараж или подвал, числовые значения должны быть равны 0.

#4. Для некоторых переменных используем наиболее частое значение
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
#Для нескольких категориальных признаков пропуски заменяем наиболее частым значением (mode).

#5. LotFrontage/Обработка отсутствующих значений
df['LotFrontage'] = df.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#Заполняем LotFrontage медианой по каждому району (Neighborhood), так как дома в одном районе имеют похожие характеристики участков.

#6. Проверка остатков пропусков
df.isnull().sum().sum()

df.isnull().sum().sort_values(ascending=False).head(30)

df['MasVnrType'] = df['MasVnrType'].fillna("None")

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

df.isnull().sum().sum()

#7. Удаление выбросов (Outliers)
df = df[df['GrLivArea'] < 4500]
#Удаляем явно аномальные дома с жилой площадью более 4500 кв. футов. Они нарушают линейные зависимости.
