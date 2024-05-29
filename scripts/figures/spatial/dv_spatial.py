# # ################################################################
# # #                   Diamond Valley Model Plot
# # ################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import scipy.stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
#
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


# Import Diamond Valley Data
df = pd.read_csv('../../machine_learning/dv_joined_ml_pumping_data.csv')
df = df[df.pumping_mm > 0]
n = df.shape[0]

df_list = []
years = range(2018, 2023)
np.random.seed(0)

ll_dict = {}
ul_dict = {}
for year in years:
    yearly_df = df[df.year == year]
    print(yearly_df.shape[0])
    q1 = np.percentile(yearly_df.pumping_net_et_ensemble_factor_annual, 25)
    q3 = np.percentile(yearly_df.pumping_net_et_ensemble_factor_annual, 75)
    iqr = q3 - q1
    ll = q1 - 1.5 * iqr
    if ll < 0:
        ll = np.min(yearly_df.pumping_net_et_ensemble_factor_annual)
    ul = q3 + 1.5 * iqr

    print(year, ll, ul)
    ll_dict[year] = ll
    ul_dict[year] = ul

median_year = np.median(years)
ll = np.round(ll_dict[median_year], 2)
ul = np.round(ul_dict[median_year], 2)

dv_data = df[df.pumping_net_et_ensemble_factor_annual < ul]
dv_data = dv_data[dv_data.pumping_net_et_ensemble_factor_annual > ll]

print('Median year-based LL/UL:', ll, ul)

et_vars = {
    'ensemble': 'OpenET Ensemble',
    # 'ssebop': 'SSEBop',
    # 'eemetric': 'eeMETRIC',
    # 'pt_jpl': 'PT-JPL',
    # 'sims': 'SIMS',
    # 'geesebal': 'geeSEBAL',
    # 'disalexi': 'ALEXI/DisALEXI'
}

for et_var in et_vars.keys():

    data_column = f'annual_net_et_{et_var}_mm'

    print(n, dv_data.shape[0])

    # Separeate X and y data
    X = np.array(dv_data.loc[:, data_column])  # Independent variable
    y = np.array(dv_data.loc[:, "pumping_mm"]) # Dependent variable

    # Number of bootstrap samples
    n_samples = 10000

    # Number of data points in the original dataset
    n_data_points = len(X)


    # Calculate confidence intervals for the slope
    final_slope = 1.2026788048873083
    final_intercept = 0


    # Calculate prediction intervals for new data points
    prediction_y_train = X * final_slope + final_intercept
    residuals_train = y - prediction_y_train

    # Calculate performance metrics
    mean_y = np.mean(y)
    r_squared = round(r2_score(y, prediction_y_train), 2)
    mae = round(mean_absolute_error(y, prediction_y_train) * 100 / mean_y, 2)
    rmse = round(mean_squared_error(y, prediction_y_train, squared=False) * 100 / mean_y, 2)
    cv = round(np.std(prediction_y_train) * 100 / np.mean(prediction_y_train), 2)

    print(f'R2={r_squared}, MAE={mae}%, RMSE={rmse}%, CV={cv}%')


   #