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
We propose 3 methods to compute PI: **LCP-RF** , **LCP-RF-G**, and **QRF-TC**. However, **QRF-TC** is used by default as
it provides the same level of accuracy as the other methods, but is faster. The implementation of LCP-RF and LCP-RF-G 
is not yet optimized.


- Assume we have trained a mean estimator (XGBRegressor) on the diabetes dataset.
```python
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

calibration_ratio = 0.5
test_ratio = 0.25

sklearn_data = load_diabetes()
X, y = sklearn_data.data, sklearn_data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=2023)
x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=calibration_ratio, random_state=2023)

model = XGBRegressor()
model.fit(x_train, y_train)
```
- To compute Predictive Intervals using **QRF-TC**. First, we need to define **ACPI** which has the same 
parameters as a classic RandomForest. The parameters of ACPI should be optimized to accurately predict the nonconformity scores of the calibration set.
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

- Then, we call the calibration method to run the training-conditional calibration.

```python 
alpha = 0.1
acpi.fit_calibration(x_cal, y_cal, nonconformity_func=None, quantile=1-alpha, only_qrf=True)
```

- You can compute the Prediction Intervals as follows.
```python 
y_lower, y_upper = acpi.predict_pi(x_test, method='qrf')

# Or the prediction sets if model is a classifier (NOT TESTED YET)
# y_pred_set = acpi.predict_pi(x_test, method='qrf')
```

## Notebooks

The notebook below show how to you use ACPI in practice.
- HOW_TO_ACPI.ipynb
