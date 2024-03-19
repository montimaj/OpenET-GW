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
df = pd.read_csv('../machine_learning/hb_joined_ml_pumping_data.csv')
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

    # Confidence level for prediction intervals (e.g., 95%)
    confidence_level = 0.95

    # Create an array to store the bootstrapped coefficients
    bootstrap_coefs = np.zeros((n_samples,))
    bootstrap_intercepts = np.zeros((n_samples,))

    # Create lists to store predictions from each bootstrap sample
    bootstrap_predictions = []
    np.random.seed(1234)

    # Perform bootstrap resampling
    for i in range(n_samples):
        # Randomly sample data points with replacement
        indices = np.random.choice(n_data_points, n_data_points, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]

        # Fit a linear regression model with intercept=0
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

        # Store the coefficient
        bootstrap_coefs[i] = model.coef_[0]
        bootstrap_intercepts[i] = model.intercept_

        # Generate predictions for new data points
        new_X = np.linspace(0, X.max(), n_data_points)  # New data points
        new_predictions_train = model.predict(X.reshape(-1, 1)).flatten()
        bootstrap_predictions.append(new_predictions_train)

    # Calculate confidence intervals for the slope
    final_slope = np.mean(bootstrap_coefs)
    final_intercept = np.mean(bootstrap_intercepts)
    print('Final intercept: ', final_intercept)
    ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
    ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

    # Calculate prediction intervals for new data points
    prediction_y = X * final_slope + final_intercept
    residuals = y - prediction_y
    pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
    pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

    # Calculate performance metrics
    mean_y = np.mean(y)
    r_squared = r2_score(y, prediction_y)
    mae = mean_absolute_error(y, prediction_y) * 100 / mean_y
    rmse = mean_squared_error(y, prediction_y, squared=False) * 100 / mean_y
    cv = np.std(prediction_y) * 100 / np.mean(prediction_y)

    # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
    sns.lineplot(x=new_X,y=final_intercept + final_slope*new_X,label="Linear Regression", color='k')
    sns.scatterplot(x=X, y=y, label='Meter Data', s=35, marker="o")
    sns.set_style(
        'white',
        rc={'xtick.bottom': True, 'ytick.left': True}
    )

    plt_text = '{:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv)
    if fit_intercept:
        plt_text = '{:.2f} + '.format(final_intercept) + plt_text
    plt_text = f'y = {plt_text}'
    # Add text
    plt.text(700, 100, plt_text, fontsize=18, color='black')

    # Add confidence intercal
    plt.fill_between(
        new_X,
        ci_upper * new_X + final_intercept,
        ci_lower * new_X + final_intercept,
        interpolate=True,
        color='red',
        alpha=0.3,
        label="95% CI"
    )
    plt.fill_between(
        new_X,
        new_X * final_slope + pi_lower + final_intercept,
        new_X * final_slope + pi_upper + final_intercept,
        interpolate=True,
        color='gray',
        alpha=0.3,
        label="95% PI"
    )

    plt.legend(fontsize=12, loc='upper left')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Net ET depth (mm)", fontsize=18)
    plt.ylabel("GP depth (mm)", fontsize=18)
    # plt.title("Diamond Valley Depths", fontsize=20)

    plt.ylim(-5, 1200)
    plt.xlim(-5, 1200)

    residuals /= np.std(residuals)

    res = scipy.stats.shapiro(residuals)
    print(res.statistic, res.pvalue)

    plt.savefig(f"with_lower_filter/hb_model_scatter_plot_{et_var}_intercept_{fit_intercept}.png", bbox_inches='tight', dpi=600)

# #
# #
# # # ################################################################
# # # #                   Diamond Valley Model Plot Volumes
# # # ################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


# Import Diamond Valley Data
df = pd.read_csv('../joined_data/hb_joined_et_pumping_data_all.csv')
df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# Outlier data
df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
df = df.loc[df['pumping_net_et_factor_annual']>0.7, :]
df = df[df["pumping_mm"] > 0]

np.random.seed(1234)
# Calculate volumes
df['annual_net_et_m3'] = df['annual_net_et_mm']*df['area_m2']/1000/1_000_000 # millions of m3
df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000/1_000_000 # millions of m3
data_column = 'annual_net_et_m3'

# Separeate X and y data
X = np.array(df.loc[:, data_column])  # Independent variable
y = np.array(df.loc[:, "pumping_m3"]) # Dependent variable
# Number of bootstrap samples
n_samples = 10000

# Number of data points in the original dataset
n_data_points = len(X)

# Confidence level for prediction intervals (e.g., 95%)
confidence_level = 0.95

# Create an array to store the bootstrapped coefficients
bootstrap_coefs = np.zeros((n_samples,))

# Create lists to store predictions from each bootstrap sample
bootstrap_predictions = []

# Perform bootstrap resampling
for i in range(n_samples):
    # Randomly sample data points with replacement
    indices = np.random.choice(n_data_points, n_data_points, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]

    # Fit a linear regression model with intercept=0
    model = LinearRegression(fit_intercept=False)
    model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

    # Store the coefficient
    bootstrap_coefs[i] = model.coef_[0]

    # Generate predictions for new data points
    new_X = np.linspace(0, X.max(), n_data_points)  # New data points
    new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
    bootstrap_predictions.append(new_predictions)

# Calculate confidence intervals for the slope
final_slope = np.mean(bootstrap_coefs)
ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# Calculate prediction intervals for new data points
prediction_y = X*final_slope
residuals = y - prediction_y
pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# Calculate performance metrics
mean_y = np.mean(y)
r_squared = r2_score(y, prediction_y)
mae = mean_absolute_error(y, prediction_y) * 100 / mean_y
rmse = mean_squared_error(y, prediction_y, squared=False) * 100 / mean_y
cv = np.std(prediction_y) * 100 / np.mean(prediction_y)

# Build Figure
plt.figure(figsize=(8, 8))

# Add data
sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
sns.lineplot(x=new_X,y=final_slope*new_X,label="Linear Regression", color='k')
sns.scatterplot(data=df, x=data_column, y='pumping_m3', label='Meter Data', s=35, marker="o")

# Add text
plt.text(0.45, 0.05, 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv),
          fontsize=18, color='black')

# Add confidence intercal
plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='red', alpha=0.3, label="95% CI")
plt.fill_between(new_X, new_X*final_slope+pi_lower, new_X*final_slope+pi_upper, interpolate=True, color='gray', alpha=0.3, label="95% PI")

plt.legend(fontsize=12, loc='upper left')

plt.ylim(-0.01, 0.9)
plt.xlim(-0.01, 0.9)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Net ET volume (millions of m$^3$)", fontsize=18)
plt.ylabel("GP volume (millions of m$^3$)", fontsize=18)



plt.savefig(r"with_lower_filter/hb_model_volume_scatter_plot.png", bbox_inches='tight', dpi=600)


