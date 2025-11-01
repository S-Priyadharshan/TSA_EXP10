# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 01-11-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

  df = pd.read_csv("/content/co2_gr_mlo.csv", comment="#")

df.head()

series=df['ann inc']

plt.plot(df['ann inc'])
plt.title("Annual Income over Time")
plt.show()

print("Mean:", series.mean())
print("Variance:", series.var())

result = adfuller(series.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("Series is stationary (no differencing needed).")
    d = 0
else:
    print("Series is NOT stationary (taking first difference).")
    d = 1
    series = series.diff().dropna()


plot_acf(series, lags=20)
plot_pacf(series, lags=20)
plt.show()

train_size = int(len(df) * 0.8)
train, test = df['ann inc'][:train_size], df['ann inc'][train_size:]

model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,12))
results = model.fit()

# model = SARIMAX(df['ann inc'], order=(2, 1, 2))
# fit = model.fit()
fit.plot_diagnostics(figsize=(10,6))
plt.show()

print(fit.summary())

plt.plot(df.index, df['ann inc'], label='Actual')
plt.plot(pred.index, pred, label='Predicted', color='orange')
plt.legend()
plt.title("SARIMA Forecast vs Actual")
plt.show()

mse = mean_squared_error(test, pred)
print("Mean Squared Error:", mse)

```

### OUTPUT:
<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/31eee0cb-5b60-40ff-b8e1-d346d226c4a4" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/f6491376-86e9-4149-88ec-2894aa053f2a" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/d570b877-032e-4892-8e84-03a778d99b69" />
<img width="844" height="547" alt="image" src="https://github.com/user-attachments/assets/5ee9a17a-a94d-4c8c-b23e-0f7512beb75c" />
<img width="850" height="584" alt="image" src="https://github.com/user-attachments/assets/0ec9f0f8-9e28-4bba-a927-cdea3f13a236" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
