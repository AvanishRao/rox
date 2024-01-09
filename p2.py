
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


california = datasets.fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target


print(df.head())


print(df.isnull().sum())


sns.pairplot(df)
plt.show()


print(df.cov())
print(df.corr())


X_train, X_test, y_train, y_test = train_test_split(df[california.feature_names], df['MedHouseVal'], test_size=0.2, random_state=42)


model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))

