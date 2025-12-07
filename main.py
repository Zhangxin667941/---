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


#Шаг 3 Предиктивное моделирование

!pip install scikit-learn

# Секция предварительной обработки данных
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# # Предположим, что df — это очищенный и обработанный DataFrame
df_encoded = pd.get_dummies(df, drop_first=True)  # Прямое кодирование категориальных переменных
# Разделение признаков и целевых переменных
X = df_encoded.drop("SalePrice", axis=1)  # Удалить целевую переменную «SalePrice»
y = df_encoded["SalePrice"]  # Целевая переменная — цена дома («SalePrice»).
# Обучающие и тестовые наборы разделены (80% обучающих, 20% тестовых).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#1. Обучение нескольких моделей машинного обучения
#Модель 1 — Линейная регрессия
from sklearn.linear_model import LinearRegression
# Создание модели линейной регрессии
lr = LinearRegression()
# Подогнать модель, используя данные обучающего набора.
lr.fit(X_train, y_train)
# Используйте модель для прогнозирования на тестовом наборе.
pred_lr = lr.predict(X_test)
# Рассчитайте среднюю квадратическую ошибку (MSE)
mse_lr = mean_squared_error(y_test, pred_lr)
# Рассчитайте среднеквадратичную ошибку (RMSE)
rmse_lr = np.sqrt(mse_lr)
# Рассчитайте оценку R²
r2_lr = r2_score(y_test, pred_lr)
# Выходные RMSE и R²
rmse_lr, r2_lr

#Модель 2 — Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred_rf)   
rmse_rf = np.sqrt(mse_rf)                    
r2_rf = r2_score(y_test, pred_rf)
rmse_rf, r2_rf

!pip install xgboost

#Модель 3 — XGBoost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, pred_xgb)
rmse_xgb, r2_xgb

#Таблица сравнения моделей
import pandas as pd

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [rmse_lr, rmse_rf, rmse_xgb],
    "R²": [r2_lr, r2_rf, r2_xgb]
})

results

#Выберите лучшую модель и выполните оптимизацию гиперпараметров (настройка гиперпараметров XGBoost).
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

param_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",   
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_model

Используйте оптимальную модель для окончательного прогноза
pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, pred_best)
rmse_best = np.sqrt(mse_best)

from sklearn.metrics import r2_score
r2_best = r2_score(y_test, pred_best)

rmse_best, r2_best
