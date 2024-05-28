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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


# Import Diamond Valley Data
# df = pd.read_csv('../joined_data/hb_joined_et_pumping_data_all.csv')
df = pd.read_csv('../../machine_learning/hb_joined_ml_pumping_data.csv')
df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
df = df[df.pumping_mm > 0]
n = df.shape[0]

ul = 1.5
ll = 0.7
df = df[df.pumping_net_et_ensemble_factor_annual < ul]
df = df[df.pumping_net_et_ensemble_factor_annual > ll]

print('Median year-based LL/UL:', ll, ul)
fit_intercept = False

print(n, df.shape[0])

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
    # Separeate X and y data
    X = np.array(df.loc[:, data_column])  # Independent variable
    y = np.array(df.loc[:, "pumping_mm"]) # Dependent variable

    # Number of bootstrap samples
    n_samples = 10000

    # Number of data points in the original dataset
    n_data_points = len(X)

    final_slope = 1.11
    final_intercept = 0
    prediction_y = X * final_slope + final_intercept
    residuals = y - prediction_y

    # Calculate performance metrics
    # Calculate performance metrics
    mean_y = np.mean(y)
    r_squared = round(r2_score(y, prediction_y), 2)
    mae = round(mean_absolute_error(y, prediction_y) * 100 / mean_y, 2)
    rmse = round(mean_squared_error(y, prediction_y, squared=False) * 100 / mean_y, 2)
    cv = round(np.std(prediction_y) * 100 / np.mean(prediction_y), 2)

    print(f'R2={r_squared}, MAE={mae}%, RMSE={rmse}%, CV={cv}%')




