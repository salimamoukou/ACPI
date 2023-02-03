## Adaptive Conformal Prediction Intervals (ACPI) By Reweighting Nonconfirmity Score

*Adaptive Conformal Prediction Intervals*  is a Python package that aims to provide 
Adaptive Predictive Intervals (PI) that better represent the uncertainty of the 
model by reweighting the nonconformal score with the learned weights of a Random Forest.
 
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
To make an all-in-one installation of all the setup for ACPI, you can run the bash script: install.sh
```
$ bash install.sh
```

## Adaptive Conformal Prediction Intervals (ACPI)
We propose 3 methods to compute PI: LCP-RF , LCP-RF-G, and QRF-TC. However, by default
we use QRF-TC as it is as accurate than the others while being more fast. The code of
LCP-RF and LCP-RF-G is not optimized yet.


**I. To compute PI using QRF-TC. First, we need to define ACPI which has the same 
parameters as a classic RandomForest, so its parameters should be optimized to predict accurately the nonconformity scores of the calibration set.**
```python
from acpi import ACPI

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acp = ACPI(model_cali=model, n_estimators=100, max_depth=20, min_node_size=10)

acp.fit(x_calibration, y_calibration)

v_calibration = acp.nonconformity_score(model.predict(x_calibration), y_calibration) 

# Optimize the RF to predict the nonconformity score
mae = mean_absolute_error(acp.predict(x_calibration), v_calibration)
```

**II. Then, we call the calibration method to run the training-conditional calibration.**

```python 
acp.fit_calibration(x_calibration, y_calibration, nonconformity_func=None, quantile=1-alpha, only_qrf=True)
```
You can use custom nonconformity score by using the argument 'nonconformity_func'. It takes a callable[[ndarray, ndarray], ndarray] that
return the nonconformity score given (predictions, y_true).

**II. Now, you can compute the Prediction Intervals**
```python 
# You can compute the prediction intervals using the code below
y_lower, y_upper = acp.predict_pi(x_test, method='qrf')
# Or the prediction sets
y_pred_set = acp.predict_pi(x_test, method='qrf')
```

## Notebooks

The notebook below show how to you use ACPI in practice.
- HOW_TO_ACPI.ipynb
