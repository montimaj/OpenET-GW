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
df = pd.read_csv('../machine_learning/dv_joined_ml_pumping_data.csv')
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
    'ssebop': 'SSEBop',
    'eemetric': 'eeMETRIC',
    'pt_jpl': 'PT-JPL',
    'sims': 'SIMS',
    'geesebal': 'geeSEBAL',
    'disalexi': 'ALEXI/DisALEXI'
}

for et_var in et_vars.keys():

    data_column = f'annual_net_et_{et_var}_mm'

    print(n, dv_data.shape[0])

    # Separeate X and y data
    X = np.array(dv_data.loc[:, data_column])  # Independent variable
    y = np.array(dv_data.loc[:, "pumping_mm"]) # Dependent variable

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,
        random_state=1234
    )


    print(x_train.size, x_test.size)
    # Number of bootstrap samples
    n_samples = 10000

    # Number of data points in the original dataset
    n_data_points = len(x_train)

    # Confidence level for prediction intervals (e.g., 95%)
    confidence_level = 0.95

    # Create an array to store the bootstrapped coefficients
    bootstrap_coefs = np.zeros((n_samples,))
    bootstrap_intercepts = np.zeros((n_samples,))

    # Create lists to store predictions from each bootstrap sample
    bootstrap_predictions_train = []
    bootstrap_predictions_test = []
    fit_intercept = False
    np.random.seed(1234)
    # Perform bootstrap resampling
    for i in range(n_samples):
        # Randomly sample data points with replacement
        indices = np.random.choice(n_data_points, n_data_points, replace=True)
        X_bootstrap = x_train[indices]
        y_bootstrap = y_train[indices]

        # Fit a linear regression model with intercept=0
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

        # Store the coefficient
        bootstrap_coefs[i] = model.coef_[0]
        bootstrap_intercepts[i] = model.intercept_

        # Generate predictions for new data points
        new_X_train = np.linspace(0, x_train.max(), n_data_points)  # New data points
        new_predictions_train = model.predict(x_train.reshape(-1, 1)).flatten()
        bootstrap_predictions_train.append(new_predictions_train)

        new_X_test = np.linspace(0, x_test.max(), n_data_points)  # New data points
        new_predictions_test = model.predict(x_test.reshape(-1, 1)).flatten()
        bootstrap_predictions_test.append(new_predictions_test)

    # Calculate confidence intervals for the slope
    final_slope = np.mean(bootstrap_coefs)
    final_intercept = np.mean(bootstrap_intercepts)

    print('Final intercept: ', final_intercept)
    ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
    ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

    # Calculate prediction intervals for new data points
    prediction_y_train = x_train * final_slope + final_intercept
    residuals_train = y_train - prediction_y_train
    pi_lower_train = np.percentile(residuals_train, (1 - confidence_level) * 100 / 2)
    pi_upper_train = np.percentile(residuals_train, 100 - (1 - confidence_level) * 100 / 2)

    # Calculate performance metrics
    mean_y = np.mean(y_train)
    r_squared = r2_score(y_train, prediction_y_train)
    mae = mean_absolute_error(y_train, prediction_y_train) * 100 / mean_y
    rmse = mean_squared_error(y_train, prediction_y_train, squared=False) * 100 / mean_y
    cv = np.std(prediction_y_train) * 100 / np.mean(prediction_y_train)


    # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    sns.lineplot(x=[0, new_X_train.max()], y= [0,new_X_train.max()], label='1:1 Line')
    sns.lineplot(x=new_X_train,y=final_intercept + final_slope*new_X_train,label="Linear Regression", color='k')
    sns.scatterplot(x=x_train, y=y_train, label='Meter Data', s=35, marker="o")
    sns.set_style(
        'white',
        rc={'xtick.bottom': True, 'ytick.left': True}
    )

    plt_text = '{:.2f}*x \n$R^2$ = {:.3f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv)
    if fit_intercept:
        plt_text = '{:.2f} + '.format(final_intercept) + plt_text
    plt_text = f'y = {plt_text}'
    # Add text
    plt.text(700, 100, plt_text, fontsize=18, color='black')

    # Add confidence intercal
    plt.fill_between(
        new_X_train,
        ci_upper * new_X_train + final_intercept,
        ci_lower * new_X_train + final_intercept,
        interpolate=True,
        color='red',
        alpha=0.3,
        label="95% CI"
    )
    plt.fill_between(
        new_X_train,
        new_X_train * final_slope + pi_lower_train + final_intercept,
        new_X_train * final_slope + pi_upper_train + final_intercept,
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

    plt.savefig(f"with_lower_filter/dv_model_scatter_plot_train_{et_var}_intercept_{fit_intercept}.png", bbox_inches='tight', dpi=600)

    residuals_train /= np.std(residuals_train)
    res = scipy.stats.shapiro(residuals_train)
    print('DV Shapiro train:', res.statistic, res.pvalue)

    print('DV train skewness:', scipy.stats.skew(residuals_train))
    print('DV train kurtosis:', scipy.stats.kurtosis(residuals_train))

    # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    iqr = np.percentile(residuals_train, 75) - np.percentile(residuals_train, 25)
    bin_width = 2 * iqr / (len(residuals_train) ** (1/3))
    bins=int((max(residuals_train) - min(residuals_train)) / bin_width)
    ax = sns.histplot(residuals_train, bins=bins, kde=True)
    ax.lines[0].set_color('crimson')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Standardized residuals (training data)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    # plt.title("Diamond Valley", fontsize=20)

    plt.savefig(f"with_lower_filter/dv_model_std_residual_histogram_plot_train_{et_var}.png", bbox_inches='tight', dpi=600)



    # Calculate prediction intervals for new data points
    prediction_y_test = x_test * final_slope + final_intercept
    residuals_test = y_test - prediction_y_test
    pi_lower_test = np.percentile(residuals_test, (1 - confidence_level) * 100 / 2)
    pi_upper_test = np.percentile(residuals_test, 100 - (1 - confidence_level) * 100 / 2)

    # Calculate performance metrics
    mean_y = np.mean(y_test)
    r_squared = r2_score(y_test, prediction_y_test)
    mae = mean_absolute_error(y_test, prediction_y_test) * 100 / mean_y
    rmse = mean_squared_error(y_test, prediction_y_test, squared=False) * 100 / mean_y
    cv = np.std(prediction_y_test) * 100 / np.mean(prediction_y_test)


    # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    sns.lineplot(x=[0, new_X_test.max()], y= [0,new_X_test.max()], label='1:1 Line')
    sns.lineplot(x=new_X_test,y=final_intercept + final_slope*new_X_test,label="Linear Regression", color='k')
    sns.scatterplot(x=x_test, y=y_test, label='Meter Data', s=35, marker="o")
    sns.set_style(
        'white',
        rc={'xtick.bottom': True, 'ytick.left': True}
    )

    plt_text = '{:.2f}*x \n$R^2$ = {:.3f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv)
    if fit_intercept:
        plt_text = '{:.2f} + '.format(final_intercept) + plt_text
    plt_text = f'y = {plt_text}'
    # Add text
    plt.text(700, 100, plt_text, fontsize=18, color='black')

    # Add confidence intercal
    plt.fill_between(
        new_X_test,
        ci_upper * new_X_test + final_intercept,
        ci_lower * new_X_test + final_intercept,
        interpolate=True,
        color='red',
        alpha=0.3,
        label="95% CI"
    )
    plt.fill_between(
        new_X_test,
        new_X_test * final_slope + pi_lower_test + final_intercept,
        new_X_test * final_slope + pi_upper_test + final_intercept,
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

    plt.savefig(f"with_lower_filter/dv_model_scatter_plot_test_{et_var}_intercept_{fit_intercept}.png", bbox_inches='tight', dpi=600)
    residuals_test /= np.std(residuals_test)
    res = scipy.stats.shapiro(residuals_test)
    print('DV Shapiro test:', res.statistic, res.pvalue)

    print('DV test skewness:', scipy.stats.skew(residuals_test))
    print('DV test kurtosis:', scipy.stats.kurtosis(residuals_test))

    # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    iqr = np.percentile(residuals_test, 75) - np.percentile(residuals_test, 25)
    bin_width = 2 * iqr / (len(residuals_test) ** (1/3))
    bins=int((max(residuals_test) - min(residuals_test)) / bin_width)
    ax = sns.histplot(residuals_test, bins=bins, kde=True)
    ax.lines[0].set_color('crimson')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Standardized residuals (test data)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    # plt.title("Diamond Valley", fontsize=20)

    plt.savefig(f"with_lower_filter/dv_model_std_residual_histogram_plot_{et_var}_test.png", bbox_inches='tight', dpi=600)


    # # Build Figure
    plt.figure(figsize=(8, 8))

    # Add data
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    sns.scatterplot(x=x_test, y=residuals_test, s=35, marker="o")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Net ET depth (mm)", fontsize=18)
    plt.ylabel("Standardized residuals (test data)", fontsize=18)
    # plt.title("Diamond Valley", fontsize=20)

    plt.savefig(f"with_lower_filter/dv_model_std_residual_scatter_plot_{et_var}_test.png", bbox_inches='tight', dpi=600)






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
df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# Outlier data
dv_data = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
dv_data = dv_data.loc[dv_data['pumping_net_et_factor_annual']>0.7, :]
dv_data = dv_data[dv_data["pumping_mm"] > 0]

# Calculate volumes
dv_data['annual_net_et_m3'] = dv_data['annual_net_et_mm']*dv_data['area_m2']/1000/1_000_000 # millions of m3
dv_data['pumping_m3'] = dv_data['pumping_mm']*dv_data['area_m2']/1000/1_000_000 # millions of m3
data_column = 'annual_net_et_m3'

# Separeate X and y data
X = np.array(dv_data.loc[:, data_column])  # Independent variable
y = np.array(dv_data.loc[:, "pumping_m3"]) # Dependent variable
# Number of bootstrap samples
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    random_state=1234
)
print(x_train.size, x_test.size)
# Number of bootstrap samples
n_samples = 10000

# Number of data points in the original dataset
n_data_points = len(x_train)

# Confidence level for prediction intervals (e.g., 95%)
confidence_level = 0.95

# Create an array to store the bootstrapped coefficients
bootstrap_coefs = np.zeros((n_samples,))
bootstrap_intercepts = np.zeros((n_samples,))

# Create lists to store predictions from each bootstrap sample
bootstrap_predictions_train = []
bootstrap_predictions_test = []
fit_intercept = False
np.random.seed(1234)
# Perform bootstrap resampling
for i in range(n_samples):
    # Randomly sample data points with replacement
    indices = np.random.choice(n_data_points, n_data_points, replace=True)
    X_bootstrap = x_train[indices]
    y_bootstrap = y_train[indices]

    # Fit a linear regression model with intercept=0
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

    # Store the coefficient
    bootstrap_coefs[i] = model.coef_[0]
    bootstrap_intercepts[i] = model.intercept_

    # Generate predictions for new data points
    new_X_train = np.linspace(0, x_train.max(), n_data_points)  # New data points
    new_predictions_train = model.predict(x_train.reshape(-1, 1)).flatten()
    bootstrap_predictions_train.append(new_predictions_train)

    new_X_test = np.linspace(0, x_test.max(), n_data_points)  # New data points
    new_predictions_test = model.predict(x_test.reshape(-1, 1)).flatten()
    bootstrap_predictions_test.append(new_predictions_test)

# Calculate confidence intervals for the slope
final_slope = np.mean(bootstrap_coefs)
final_intercept = np.mean(bootstrap_intercepts)

print('Final intercept: ', final_intercept)
ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# Calculate prediction intervals for new data points
prediction_y_train = x_train * final_slope + final_intercept
residuals_train = y_train - prediction_y_train
pi_lower_train = np.percentile(residuals_train, (1 - confidence_level) * 100 / 2)
pi_upper_train = np.percentile(residuals_train, 100 - (1 - confidence_level) * 100 / 2)

# Calculate performance metrics
mean_y = np.mean(y_train)
r_squared = r2_score(y_train, prediction_y_train)
mae = mean_absolute_error(y_train, prediction_y_train) * 100 / mean_y
rmse = mean_squared_error(y_train, prediction_y_train, squared=False) * 100 / mean_y
cv = np.std(prediction_y_train) * 100 / np.mean(prediction_y_train)

# Build Figure
plt.figure(figsize=(8, 8))

# Add data
sns.lineplot(x=[0, new_X_train.max()], y= [0,new_X_train.max()], label='1:1 Line')
sns.lineplot(x=new_X_train,y=final_intercept + final_slope*new_X_train,label="Linear Regression", color='k')
sns.scatterplot(x=x_train, y=y_train, label='Meter Data', s=35, marker="o")
sns.set_style(
    'white',
    rc={'xtick.bottom': True, 'ytick.left': True}
)

plt_text = '{:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv)
if fit_intercept:
    plt_text = '{:.2f} + '.format(final_intercept) + plt_text
plt_text = f'y = {plt_text}'
# Add text
plt.text(1.1, 0.1, plt_text, fontsize=18, color='black')

# Add confidence intercal
plt.fill_between(
    new_X_train,
    ci_upper * new_X_train + final_intercept,
    ci_lower * new_X_train + final_intercept,
    interpolate=True,
    color='red',
    alpha=0.3,
    label="95% CI"
)
plt.fill_between(
    new_X_train,
    new_X_train * final_slope + pi_lower_train + final_intercept,
    new_X_train * final_slope + pi_upper_train + final_intercept,
    interpolate=True,
    color='gray',
    alpha=0.3,
    label="95% PI"
)

plt.legend(fontsize=12, loc='upper left')
plt.ylim(0, 2.1)
plt.xlim(0, 2.1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Net ET volume (millions of m$^3$)", fontsize=18)
plt.ylabel("GP volume (millions of m$^3$)", fontsize=18)
# plt.title("Diamond Valley Depths", fontsize=20)



plt.savefig(f"with_lower_filter/dv_model_volume_scatter_plot_train_intercept_{fit_intercept}.png", bbox_inches='tight', dpi=600)


# Calculate prediction intervals for new data points
prediction_y_test = x_test * final_slope + final_intercept
residuals_test = y_test - prediction_y_test
pi_lower_test = np.percentile(residuals_test, (1 - confidence_level) * 100 / 2)
pi_upper_test = np.percentile(residuals_test, 100 - (1 - confidence_level) * 100 / 2)

# Calculate performance metrics
mean_y = np.mean(y_test)
r_squared = r2_score(y_test, prediction_y_test)
mae = mean_absolute_error(y_test, prediction_y_test) * 100 / mean_y
rmse = mean_squared_error(y_test, prediction_y_test, squared=False) * 100 / mean_y
cv = np.std(prediction_y_test) * 100 / np.mean(prediction_y_test)


# Build Figure
plt.figure(figsize=(8, 8))

# Add data
sns.lineplot(x=[0, new_X_test.max()], y= [0,new_X_test.max()], label='1:1 Line')
sns.lineplot(x=new_X_test,y=final_intercept + final_slope*new_X_test,label="Linear Regression", color='k')
sns.scatterplot(x=x_test, y=y_test, label='Meter Data', s=35, marker="o")
sns.set_style(
    'white',
    rc={'xtick.bottom': True, 'ytick.left': True}
)

plt_text = '{:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(final_slope, r_squared, rmse,mae, cv)
if fit_intercept:
    plt_text = '{:.2f} + '.format(final_intercept) + plt_text
plt_text = f'y = {plt_text}'
# Add text
plt.text(1.1, 0.1, plt_text, fontsize=18, color='black')

# Add confidence intercal
plt.fill_between(
    new_X_test,
    ci_upper * new_X_test + final_intercept,
    ci_lower * new_X_test + final_intercept,
    interpolate=True,
    color='red',
    alpha=0.3,
    label="95% CI"
)
plt.fill_between(
    new_X_test,
    new_X_test * final_slope + pi_lower_test + final_intercept,
    new_X_test * final_slope + pi_upper_test + final_intercept,
    interpolate=True,
    color='gray',
    alpha=0.3,
    label="95% PI"
)

plt.legend(fontsize=12, loc='upper left')
plt.ylim(0, 2.1)
plt.xlim(0, 2.1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Net ET volume (millions of m$^3$)", fontsize=18)
plt.ylabel("GP volume (millions of m$^3$)", fontsize=18)
# plt.title("Diamond Valley Depths", fontsize=20)



plt.savefig(f"with_lower_filter/dv_model_volume_scatter_plot_test_intercept_{fit_intercept}.png", bbox_inches='tight', dpi=600)




#
#
# # ################################################################
# # #                   Diamond Valley Predicted Pumping
# # ################################################################
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# Import et data
df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')
df['est_pumping_mm'] = df['annual_net_et_mm']*1.12
df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000
df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]
# df = df[df["pumping_mm"] > 0]


# Get estimated data
et_df = df.loc[:, ['year',]].copy()
et_df['data'] = df['est_pumping_m3']
et_df = et_df.groupby('year').sum()
et_df = et_df.reset_index()
et_df['dataset'] = 'model'

# Get estimated data
pumping_df = df.loc[:, ['year',]].copy()
pumping_df['data'] = df['pumping_m3']
pumping_df = pumping_df.groupby('year').sum()
pumping_df = pumping_df.reset_index()
pumping_df['dataset'] = 'actual'

# combine data
df_plot = pd.concat([et_df, pumping_df]).sort_values(by=['year', 'data'])
df_plot['year'] = df_plot['year'].astype(int)
df_plot['data'] = df_plot['data']/1_000_000

# Sample data structure (replace this with your actual data)
data = {
    'Year': df_plot.year,
    'Category': df_plot.dataset,
    'Value': df_plot.data
}

df = pd.DataFrame(data)

# Create a barplot using Seaborn
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})
sns.set_theme(style="whitegrid")

# Replace 'Value' and 'Category' with your actual column names
ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')

ax.bar_label(ax.containers[0], fmt='%.1f')
ax.bar_label(ax.containers[1], fmt='%.1f')
# Set plot labels and title
plt.xlabel('Year')
plt.ylabel('GP volume (Mm$^3$)')
plt.title('Total vs Modeled Withdrawals Diamond Valley')

plt.ylim(0, 95)

# Show the legend
plt.legend()

plt.savefig('with_lower_filter/dv_bar_plot.png', bbox_inches='tight', dpi=400)
#