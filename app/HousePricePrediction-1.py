# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import sklearn
print(sklearn.__version__)

# %%
df = pd.read_csv('../data/House Price India.csv')

# %%
df.head()

# %%
df.shape

# %%
df.columns

# %%
df.isnull().sum()

# %%
df.duplicated().any()

# %%
corr_matrix = df.corr()

# %%
corr_matrix

# %%
corr_with_price = corr_matrix['Price'].sort_values(ascending=False)
corr_with_price

# %%
X = df.drop(columns=['Price', 'id', 'Date', 'Postal Code'])
y = df['Price']

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# %%
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

# %%
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# %%
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# %%
joblib.dump(rf, "../models/random_forest_model.pkl")

# %%
X_train.columns

# %%


# %%



