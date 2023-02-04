![build and test](https://github.com/salimamoukou/ACPI/actions/workflows/build_test.yml/badge.svg)
## Adaptive Conformal Prediction Intervals (ACPI)

**ACPI** offers Adaptive Predictive Intervals (PI) that accurately reflect the
 uncertainty of a given model, with finite-sample marginal and training-conditional coverage, 
 as well as asymptotic conditional coverage.  It has been proven to significantly improve upon the split-conformal 
 approach, regardless of the nonconformity score used (mean regression, quantile regression, etc.).
## Requirements
Python 3.7+ 

**OSX**: ACPI uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

Install the required packages:

```
$ pip install -r requirements.txt
```

## Installation

Clone the repo and run the following command in the ACPI directory to install ACPI
```
$ pip install .
```
To make an all-in-one installation, you can run the bash script: install.sh
```
$ bash install.sh
```

## Quickstart
We propose 3 methods to compute PI: **LCP-RF** , **LCP-RF-G**, and **QRF-TC**.
- **LCP-RF**: Random Forest Localizer. It used the learned weights of the RF to give more importance to calibration 
samples that have residuals similar to the test points in the calibration step.

- **LCP-RF-G**: Groupwise-Random Forest Localizer. It extends the previous approach by conformalizing by group. The groups
are computed using the weights of the forest that permits to find cluster/partition of the space with similar residuals.
Hence, allowing more efficient and more adaptive Predictive Intervals.

- **QRF-TC**: It directly calibrates the Random Forest Localizer to achieve training-conditional coverage.
 
**Remarks**. We used **QRF-TC** by default as it provides the same level of accuracy as the other methods, but is faster. 

- **Assume we have trained a mean estimator (XGBRegressor) on the california house prices dataset.**
```python
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split

calibration_ratio = 0.5
test_ratio = 0.25

sklearn_data = fetch_california_housing()
X, y = sklearn_data.data, sklearn_data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=2023)
x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=calibration_ratio, random_state=2023)

model = XGBRegressor()
model.fit(x_train, y_train)
```
- **To compute Predictive Intervals using **QRF-TC**. First, we need to define **ACPI** which has the same 
parameters as a classic RandomForest. The parameters of ACPI should be optimized to accurately predict the nonconformity 
scores of the calibration set.**
```python
from sklearn.metrics import mean_absolute_error
from acpi import ACPI

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acpi = ACPI(model_cali=model, n_estimators=100, max_depth=20, min_node_size=10)

acpi.fit(x_cal, y_cal, nonconformity_func=None)
# You can use custom nonconformity score by using the argument 'nonconformity_func'. 
# It takes a callable[[ndarray, ndarray], ndarray] that return the nonconformity 
# score given (predictions, y_true). By the default, it uses absolute residual if model 
# is mean estimator and max(pred_lower - y, y - pred_upper) if the model is quantile estimates.

v_cal = acpi.nonconformity_score(model.predict(x_cal), y_cal) 

# Optimize the RF to predict the nonconformity score
mae = mean_absolute_error(acpi.predict(x_cal), v_cal)
```

- **Then, we call the calibration method to run the training-conditional calibration.**

```python 
alpha = 0.1
acpi.fit_calibration(x_cal, y_cal, nonconformity_func=None, quantile=1-alpha, only_qrf=True)
```

- **You can compute the Prediction Intervals as follows.**
```python 
y_lower, y_upper = acpi.predict_pi(x_test, method='qrf')

# Or the prediction sets if model is a classifier (NOT TESTED YET)
# y_pred_set = acpi.predict_pi(x_test, method='qrf')
```

### Improvements over split-CP
- For the sake of demonstration, we compare the intervals width given by our methods with split-CP in the previous 
datasets. We used the library MAPIE to compute split-CP PI.
```python 
from mapie.regression import MapieRegressor

mapie = MapieRegressor(model, method='base', cv='prefit')
mapie.fit(x_cal, y_cal)
y_test_pred, y_test_pis = mapie.predict(x_test, alpha=alpha)
```
- Below, we plot the interval width of split-CP, QRF-TC and the true errors of the model. It shows that our methods
given varied interval (second figure) while split-CP PI are constant. The figure 1 and 3 shows that our PI
are more aligned with the true errors of the model.
```python 
idx = list(range(len(y_test)))
sort_id = np.argsort(y_test)
max_size = 500

y_lower_true = model.predict(x_test) - np.abs(model.predict(x_test) - y_test)
y_upper_true = model.predict(x_test) + np.abs(model.predict(x_test) - y_test)
r = {}
r['QRF_TC'] = y_upper - y_lower
r['True'] = y_upper_true - y_lower_true
r['SPLIT'] = y_test_pis[:, 1, 0] - y_test_pis[:, 0, 0]

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].errorbar(idx[:max_size], y_test[sort_id][:max_size], yerr=r['SPLIT'][sort_id][:max_size], fmt='o', label='SPLIT', color='tab:orange')
ax[0].errorbar(idx[:max_size], y_test[sort_id][:max_size], yerr=r['QRF_TC'][sort_id][:max_size], fmt='o', label='QRF_TC', color='tab:blue')
ax[0].errorbar(idx[:max_size], y_test[sort_id][:max_size], yerr=r['True'][sort_id][:max_size], fmt='o', label='True errors', color='tab:green')
ax[0].set_ylabel('Interval width')
ax[0].set_xlabel('Order of True values')
ax[0].legend()

ax[1].scatter(y_test, r['QRF_TC'], label='QRF_TC')
ax[1].scatter(y_test, r['SPLIT'], label='SPLIT')
ax[1].set_xlabel("True values", fontsize=12)
ax[1].set_ylabel("Interval width", fontsize=12)
ax[1].set_xscale("linear")
ax[1].set_ylim([0, np.max(r['QRF_TC'])*1.1])
ax[1].legend()



ax[2].scatter(y_test, r['True'], label='True errors', color='tab:green')
ax[2].scatter(y_test, r['QRF_TC'], label='QRF_TC', color='tab:blue')
ax[2].scatter(y_test, r['SPLIT'], label='SPLIT', color='tab:orange')
ax[2].set_xlabel("True values", fontsize=12)
ax[2].set_ylabel("Interval width", fontsize=12)
ax[2].set_xscale("linear")
ax[2].set_ylim([0, np.max(r['QRF_TC'])*1.1])
ax[2].legend()

plt.suptitle('Intervals width comparisons between SPLIT, QRF-TC, and the True error ', size=20)
plt.show()
```


## Notebooks

The notebook below show how to you use ACPI for quantile regression and mean regression.
- HOW_TO_ACPI.ipynb

## Improvement over split-conformal approaches
