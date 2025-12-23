import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction – MLR",
    layout="wide"
)

st.title("🏠 House Price Prediction using Multiple Linear Regression")
st.markdown("This application demonstrates **Multiple Linear Regression with Backward Elimination**.")

# ---------------- LOAD MODEL ----------------
model_path = "Multiple_Linear_regression_house_price.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance")

bias = 0.706       # replace with exact values if needed
variance = 0.685   # replace with exact values if needed

col1, col2 = st.columns(2)
col1.metric("Training Score (Bias)", round(bias, 3))
col2.metric("Testing Score (Variance)", round(variance, 3))

# ---------------- DATASET  ----------------
st.subheader("📁 Dataset ")

data = {
    "id": [7129300520, 6414100192, 5631500400, 2487200875, 1954400510],
    "date": ["20141013T000000", "20141209T000000", "20150225T000000",
             "20141209T000000", "20150218T000000"],
    "price": [221900.0, 538000.0, 180000.0, 604000.0, 510000.0],
    "long": [-122.257, -122.319, -122.233, -122.393, -122.045],
    "sqft_living15": [1340, 1690, 2720, 1360, 1800],
    "sqft_lot15": [5650, 7639, 8062, 5000, 7503]
}

df_sample = pd.DataFrame(data)
st.dataframe(df_sample, use_container_width=True)

st.markdown("""
📌 **Note:**  
- This is a sample preview of the original House Price dataset  
- The full dataset contains **21,613 records** ** 21 features**
- Used for **Multiple Linear Regression with Backward Elimination**
""")



# ---------------- OLS SUMMARY ----------------
st.subheader("📄 OLS Regression Summary")

ols_summary = """
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.905
Model:                            OLS   Adj. R-squared (uncentered):              0.905
Method:                 Least Squares   F-statistic:                          1.287e+04
Date:                Mon, 22 Dec 2025   Prob (F-statistic):                        0.00
Time:                        11:24:13   Log-Likelihood:                     -2.9461e+05
No. Observations:               21613   AIC:                                  5.892e+05
Df Residuals:                   21597   BIC:                                  5.894e+05
Df Model:                          16                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1         -3.562e+04   1887.390    -18.872      0.000   -3.93e+04   -3.19e+04
x2          4.234e+04   3136.467     13.500      0.000    3.62e+04    4.85e+04
x3           109.9055      2.256     48.724      0.000     105.484     114.327
x4             0.1312      0.048      2.742      0.006       0.037       0.225
x5          5.833e+05   1.74e+04     33.600      0.000    5.49e+05    6.17e+05
x6          5.249e+04   2126.373     24.686      0.000    4.83e+04    5.67e+04
x7          2.691e+04   2315.359     11.624      0.000    2.24e+04    3.15e+04
x8          9.581e+04   2133.801     44.903      0.000    9.16e+04       1e+05
x9            72.5890      2.088     34.763      0.000      68.496      76.682
x10           37.3165      2.407     15.506      0.000      32.599      42.033
x11        -2544.6464     67.021    -37.968      0.000   -2676.013   -2413.280
x12           20.6412      3.643      5.666      0.000      13.500      27.782
x13         -521.7152     17.738    -29.413      0.000    -556.482    -486.948
x14         6.036e+05   1.07e+04     56.456      0.000    5.83e+05    6.25e+05
x15        -2.192e+05    1.3e+04    -16.824      0.000   -2.45e+05   -1.94e+05
x16           22.3571      3.355      6.664      0.000      15.782      28.932
x17           -0.3807      0.073     -5.204      0.000      -0.524      -0.237
==============================================================================
Omnibus:                    18359.519   Durbin-Watson:                   1.991
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1857062.968
Skew:                           3.560   Prob(JB):                         0.00
Kurtosis:                      47.850   Cond. No.                     2.52e+17
==============================================================================
Notes:
- Model does not include intercept
- High multicollinearity detected
"""

st.code(ols_summary, language="text")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("📌 **MLR Project | Backward Elimination | StatsModels + Scikit-learn**")
