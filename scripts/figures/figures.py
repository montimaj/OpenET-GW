# ##############################################################
# #                  Diamond Valley Precip Plot
# ##############################################################

# # NIWR, ETo, ETact, and PPT

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/diamond_valley/effective_precip_fraction.csv')

# # Sample data for two datasets
# eto = df['ETpot'].values
# niwr = df['NIWR'].values
# eta = df['ETact'].values
# ppt = df['PPT'].values

# # Define the x-axis labels for each year
# years = ['2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(eto)), y=eto, color='ForestGreen', label='ASCE Grass Reference ET', linestyle=':')
# sns.scatterplot(x=range(len(eto)), y=eto, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(eta)), y=eta, color='black', label='Modeled Actual ET', linestyle='-.')
# sns.scatterplot(x=range(len(eta)), y=eta, color='black', marker='o')

# sns.lineplot(x=range(len(niwr)), y=niwr, color='SteelBlue', label='Net Irrigation Water Requirement', linestyle='--')
# sns.scatterplot(x=range(len(niwr)), y=niwr, color='SteelBlue', marker='o')

# sns.lineplot(x=range(len(ppt)), y=ppt, color='FireBrick', label='Total Precipitation', linestyle='-')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis

# # Set y limits
# ax1.set_ylim(0, 1250)  # Set the y-axis limits for Dataset 1
# ax1.set_ylim(0, 1250) 

# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.set_ylabel('Flux (mm)')

# # ax1.legend(loc='lower left')
# ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))


# plt.title('Diamond Valley')

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/dv_et_demands_plot.png', dpi=300)

# # Show the plot
# plt.show()


# ##############################################################
# #                  Diamond Valley Root Zone Plot
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/diamond_valley/effective_precip_fraction.csv')

# # Sample data for two datasets
# ppt = df['PPT'].values
# p_rz = df['P_rz'].values
# p_rz_fraction = df['P_rz_fraction'].values

# # Define the x-axis labels for each year
# years = ['2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(ppt)), y=ppt, color='ForestGreen', label='Total Precipitation', linestyle=':')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', label='Effective Precipitation', linestyle='--')
# sns.scatterplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', marker='o')

# # Create the secondary axis
# ax2 = ax1.twinx()
# sns.lineplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', ax=ax2, label='Fraction of Effective Precipitation')
# sns.scatterplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis
# ax2.grid(False)  # Remove grid lines on the secondary axis

# # Set y limits
# ax1.set_ylim(0, 355)  # Set the y-axis limits for Dataset 1
# ax2.set_ylim(0.5, 1.01)  # Set the y-axis limits for Dataset 1

# # Set labels and legends for both axes
# ax1.set_ylabel('Flux (mm)')
# ax2.set_ylabel('Fraction')
# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.legend(loc='lower left')
# ax2.legend(loc='lower right')

# plt.title('Diamond Valley')

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/dv_precip_plot.png', dpi=300)


# # Show the plot
# plt.show()

# ##############################################################
# #                  Diamond Valley Precip Plot EFF
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/diamond_valley/effective_precip_fraction.csv')

# # Sample data for two datasets
# ppt = df['PPT'].values
# p_rz = df['P_eft'].values
# p_rz_fraction = df['P_eft_fraction'].values

# # Define the x-axis labels for each year
# years = ['2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(ppt)), y=ppt, color='ForestGreen', label='Total Precipitation', linestyle=':')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', label='P_eft', linestyle='--')
# sns.scatterplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', marker='o')

# # Create the secondary axis
# ax2 = ax1.twinx()
# sns.lineplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', ax=ax2, label='P_eft_fraction')
# sns.scatterplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis
# ax2.grid(False)  # Remove grid lines on the secondary axis

# # Set y limits
# ax1.set_ylim(0, 355)  # Set the y-axis limits for Dataset 1
# ax2.set_ylim(0.5, 1.01)  # Set the y-axis limits for Dataset 1

# # Set labels and legends for both axes
# ax1.set_ylabel('Flux (mm)')
# ax2.set_ylabel('Fraction')
# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.legend(loc='lower left')
# ax2.legend(loc='lower right')

# plt.title('Diamond Valley')

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/dv_peft_.png', dpi=300)


# # Show the plot
# plt.show()


# ##############################################################
# #                  Diamond Valley Box Plot
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # REmove factor
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]
# df = df[df["pumping_mm"] > 0]

# ##### Build Box Plot #####
# # Reformat dataframe for plotting
# df1 = df.loc[:, ['year', 'annual_net_et_mm']].rename(columns={'annual_net_et_mm': 'data'})
# df1['dataset'] = 'Net ET'

# df2 = df.loc[:, ['year', 'pumping_mm']].rename(columns={'pumping_mm': 'data'})
# df2['dataset'] = 'Pumping'

# # df3 = df.loc[:, ['year', 'annual_precip_eff_mm']].rename(columns={'annual_precip_eff_mm': 'data'})
# # df3['dataset'] = 'Effective Precipitation'

# df_plot = pd.concat([df1, df2], axis=0)

# # Sample data as Pandas DataFrames (replace this with your data)
# data = pd.DataFrame({
#     'Year': df_plot.year,
#     'Dataset': df_plot.dataset,
#     'Data': df_plot.data
# })

# # Create a publication-quality plot
# sns.set(style="whitegrid", font_scale=1.2)  # Set the plot style and increase font size
# plt.figure(figsize=(10, 6))
# plt.ylim(0, 1400)

# # Define your custom color palette
# colors = {"Net ET": "#4682B4", "Pumping": "#228B22"}

# # Create the box plot
# ax = sns.boxplot(data=data, x='Year', y='Data', hue='Dataset',
#                   width=0.6, palette=colors, dodge=True)

# # Customize plot appearance
# ax.set_xlabel('Year', fontsize=14)
# ax.set_ylabel('Depth (mm)', fontsize=14)
# ax.legend(loc='upper left') # (title='Dataset', fontsize=12, title_fontsize=12)

# # Customize tick label font size
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)

# # Add grid lines
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Remove top and right spines
# sns.despine()

# # Set the title and adjust its font size
# ax.set_title('Annual Pumping and Net ET Diamond Valley', fontsize=16)

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/dv_box_plot.png', dpi=300)

# # Show the plot
# plt.show()


# ################################################################
# #                   Diamond Valley Outlier
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # Meter df
# outlier_df = df.loc[(df['pumping_net_et_factor_annual']>1.5) |
#                     # (df['pumping_net_et_factor_annual']<0.5) |
#                     (df["pumping_mm"] == 0), :]

# data_df = df.loc[(df['pumping_net_et_factor_annual']<1.5) &
#                   # (df['pumping_net_et_factor_annual']>0.5) &
#                   (df["pumping_mm"] > 0), :]

# # outlier_df = df.loc[df['pumping_net_et_factor_annual']>1.5, :]
# # data_df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]

# plt.figure(figsize=(8, 8))

# # Set the style to 'whitegrid' or 'ticks' to add gridlines
# sns.set_style('ticks')  # Use 'ticks' for gridlines with ticks

# # Create a scatter plot using DataFrame with different symbols for each dataset
# sns.scatterplot(data=data_df, x='annual_net_et_mm', y='pumping_mm', label='Data', s=50, marker="o")
# sns.scatterplot(data=outlier_df, x='annual_net_et_mm', y='pumping_mm', label='Outlier', s=50, marker="^")

# # Add horizontal line
# plt.axvline(x=800, color='grey', linestyle='--', label='NIWR: 800mm')
# plt.axhline(y=940, color='grey', linestyle='-.', label='NIWR/0.85: 940mm')

# # Adding labels and title with larger fonts
# plt.xlabel("Net ET Flux (mm)", fontsize=18)
# plt.ylabel("Pumping Flux (mm)", fontsize=18)
# plt.title("Scatter Plot of Outliers Diamond Valley", fontsize=20)

# # Customize tick label fonts
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Set limit
# plt.ylim(-100, 2800) 
# # plt.xlim(-100, 2800) 


# # Adding a legend with larger fonts
# plt.legend(fontsize=16, loc='upper left')

# # Customize the marker colors
# sns.set_palette("colorblind")  # Use a colorblind-friendly palette

# # Save the plot as a high-resolution image (e.g., PNG)
# plt.savefig(r"without_lower_filter/dv_outlier_scatter_plot.png", dpi=400)

# # Show the plot
# plt.show()


# ################################################################
# #                   Diamond Valley Model Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from matplotlib import pyplot as plt


# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # Outlier data
# dv_data = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # dv_data = dv_data.loc[df['pumping_net_et_factor_annual']>0.5, :]
# dv_data = dv_data[dv_data["pumping_mm"] > 0]
# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(dv_data.loc[:, data_column])  # Independent variable
# y = np.array(dv_data.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate performance metrics
# r_squared = r2_score(y, X * final_slope)
# mae = mean_absolute_error(y, X * final_slope) * 100 / np.mean(y)
# mse = mean_squared_error(y, X * final_slope)
# rmse = np.sqrt(mse) * 100 / np.mean(y)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
# sns.lineplot(x=new_X,y=final_slope*new_X,label="Linear Regression")
# sns.scatterplot(data=dv_data, x=data_column, y='pumping_mm', label='Meter Data', s=35, marker="o")
# sns.set(style="white")

# # Add text
# plt.text(800, 100, 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}%'.format(final_slope, r_squared, rmse ,mae),
#           fontsize=18, color='black')

# # Add confidence intercal
# plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3, label="95% CI")
# plt.fill_between(new_X, new_X*final_slope+pi_lower, new_X*final_slope+pi_upper, interpolate=True, color='gray', alpha=0.3, label="95% PI")

# plt.legend(fontsize=12, loc='upper left')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET Flux (mm)", fontsize=18)
# plt.ylabel("Pumping Flux (mm)", fontsize=18)
# plt.title("Diamond Valley", fontsize=20)

# plt.ylim(-5, 1200)
# plt.xlim(-5, 1200)

# plt.savefig(r"without_lower_filter/dv_model_scatter_plot_new.png", dpi=400)


# ################################################################
# #                   Diamond Valley Model Plot Volumes
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from matplotlib import pyplot as plt


# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # Outlier data
# dv_data = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # dv_data = dv_data.loc[df['pumping_net_et_factor_annual']>0.5, :]
# dv_data = dv_data[dv_data["pumping_mm"] > 0]

# # Calculate volumes
# dv_data['annual_net_et_m3'] = dv_data['annual_net_et_mm']*dv_data['area_m2']/1000/1_000_000 # millions of m3
# dv_data['pumping_m3'] = dv_data['pumping_mm']*dv_data['area_m2']/1000/1_000_000 # millions of m3
# data_column = 'annual_net_et_m3'

# # Separeate X and y data
# X = np.array(dv_data.loc[:, data_column])  # Independent variable
# y = np.array(dv_data.loc[:, "pumping_m3"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate performance metrics
# r_squared = r2_score(y, X * final_slope)
# mae = mean_absolute_error(y, X * final_slope) * 100 / np.mean(y)
# mse = mean_squared_error(y, X * final_slope)
# rmse = np.sqrt(mse) * 100 / np.mean(y)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
# sns.lineplot(x=new_X,y=final_slope*new_X,label="Linear Regression")
# sns.scatterplot(data=dv_data, x=data_column, y='pumping_m3', label='Meter Data', s=35, marker="o")
# sns.set(style="white")

# # Add text
# plt.text(1.1, 0.1, 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}%'.format(final_slope, r_squared, rmse ,mae),
#           fontsize=18, color='black')

# # Add confidence intercal
# plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3, label="95% CI")
# plt.fill_between(new_X, new_X*final_slope+pi_lower, new_X*final_slope+pi_upper, interpolate=True, color='gray', alpha=0.3, label="95% PI")

# plt.legend(fontsize=12, loc='upper left')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET Volume (millions of m3)", fontsize=18)
# plt.ylabel("Pumping Volume (millions of m3)", fontsize=18)
# plt.title("Diamond Valley", fontsize=20)

# plt.ylim(-0.1, 1.75)
# plt.xlim(-0.1, 1.75)

# plt.savefig(r"without_lower_filter/dv_model_volume_scatter_plot.png", dpi=400)


# ###############################################################
# #                   Diamond Valley Residule Histogram Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from matplotlib import pyplot as plt


# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # Outlier data
# dv_data = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # dv_data = dv_data.loc[df['pumping_net_et_factor_annual']>0.5, :]
# dv_data = dv_data[dv_data["pumping_mm"] > 0]
# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(dv_data.loc[:, data_column])  # Independent variable
# y = np.array(dv_data.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
# bin_width = 2 * iqr / (len(residuals) ** (1/3))
# bins=int((max(residuals) - min(residuals)) / bin_width)
# sns.histplot(residuals, bins=bins, kde=False, color='skyblue')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Residuals (mm)", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title("Diamond Valley", fontsize=20)

# plt.savefig(r"without_lower_filter/dv_model_residule_histogram_plot.png", dpi=400)


# ###############################################################
# #                   Diamond Valley Residule vs NetET Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from matplotlib import pyplot as plt


# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')

# # Outlier data
# dv_data = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # dv_data = dv_data.loc[df['pumping_net_et_factor_annual']>0.5, :]
# dv_data = dv_data[dv_data["pumping_mm"] > 0]
# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(dv_data.loc[:, data_column])  # Independent variable
# y = np.array(dv_data.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# sns.scatterplot(x=X, y=residuals, s=35, marker="o")
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET (mm)", fontsize=18)
# plt.ylabel("Residuals", fontsize=18)
# plt.title("Diamond Valley", fontsize=20)

# plt.savefig(r"without_lower_filter/dv_model_residule_scatter_plot.png", dpi=400)


# ################################################################
# #                   Diamond Valley Prediceted Pumping
# ################################################################

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Import et data
# df = pd.read_csv(r'../joined_data/dv_joined_et_pumping_data_all.csv')
# df['est_pumping_mm'] = df['annual_net_et_mm']*1.03
# df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000
# df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]
# df = df[df["pumping_mm"] > 0]


# # Get estimated data
# et_df = df.loc[:, ['year',]].copy()
# et_df['data'] = df['est_pumping_m3']
# et_df = et_df.groupby('year').sum()
# et_df = et_df.reset_index()
# et_df['dataset'] = 'model'

# # Get estimated data
# pumping_df = df.loc[:, ['year',]].copy()
# pumping_df['data'] = df['pumping_m3']
# pumping_df = pumping_df.groupby('year').sum()
# pumping_df = pumping_df.reset_index()
# pumping_df['dataset'] = 'actual'

# # combine data
# df_plot = pd.concat([et_df, pumping_df]).sort_values(by=['year', 'data'])
# df_plot['year'] = df_plot['year'].astype(int)
# df_plot['data'] = df_plot['data']/1_000_000

# # Sample data structure (replace this with your actual data)
# data = {
#     'Year': df_plot.year,
#     'Category': df_plot.dataset,
#     'Value': df_plot.data
# }

# df = pd.DataFrame(data)

# # Create a barplot using Seaborn
# plt.figure(figsize=(10, 6))
# sns.set_theme(style="whitegrid")

# # Replace 'Value' and 'Category' with your actual column names
# ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')

# for p in ax.patches:
#     ax.annotate(round(p.get_height(), 1), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
#                 textcoords='offset points')

# # Set plot labels and title
# plt.xlabel('Year')
# plt.ylabel('volume (Mm3)')
# plt.title('Total Pumping vs Modeled Pumping Diamond Valley')

# plt.ylim(0, 95)

# # Show the legend
# plt.legend()

# plt.savefig(r'without_lower_filter/dv_bar_plot.png', dpi=300)

# plt.show()


# ################################################################
# #                   Diamond Valley Prediceted Pumping (Whole Basin) Imperial
# ################################################################
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Import et data
# df = pd.read_csv(r'../openet_data/dv_openet_data_2018_2022_all.csv')
# df['annual_net_et_mm'] = df['annual_et_mm']-df['annual_precip_eff_mm']
# df = df.loc[df['annual_net_et_mm'] > 0, :] # filter out areas with less thatn 10 acres
# df['est_pumping_mm'] = df['annual_net_et_mm']*1.1142832550149728
# df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000
#
# # Get estimated data
# et_df = df.loc[:, ['YEAR',]].copy()
# et_df['data'] = df['est_pumping_m3']
# et_df = et_df.groupby('YEAR').sum()
# et_df = et_df.reset_index()
# et_df['dataset'] = 'model'
#
# # Get estimated data
# pumping_df = et_df.copy()
# pumping_df['data'] = [75297823, 68638385.42, 88459789.77, 77087599.33, 78900508.69]
# pumping_df['dataset'] = 'Actual'
#
# # Convert to acre-ft
#
# # combine data
# df_plot = pd.concat([et_df, pumping_df]).sort_values(by=['YEAR', 'data'])
# df_plot['data'] = df_plot['data']* 0.000810714/1000
# df_plot['YEAR'] = df_plot['YEAR'].astype(int)
# df_plot = df_plot.rename(columns={'YEAR': 'year'})
#
# # Sample data structure (replace this with your actual data)
# data = {
#     'Year': df_plot.year,
#     'Category': df_plot.dataset,
#     'Value': df_plot.data
# }
#
# df = pd.DataFrame(data)
#
# # Create a barplot using Seaborn
# plt.figure(figsize=(10, 6))
# sns.set_theme(style="whitegrid")
#
# # Replace 'Value' and 'Category' with your actual column names
# ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')
#
#
# for p in ax.patches[0:10]:
#     ax.annotate(round(p.get_height(), 1), (p.get_x() + p.get_width() / 2., p.get_height()+1),
#                 ha='center', va='center', fontsize=14, color='black', xytext=(0, 5),
#                 textcoords='offset points')
#
# mae = []
# for index in range(0, 5):
#     model = ax.patches[index].get_height()
#     actual = ax.patches[index+5].get_height()
#     percent =  round((actual - model)*100/actual)
#     p = ax.patches[index]
#
#     ax.annotate(f'{percent}%', (p.get_x() + p.get_width() / 2., p.get_height()+1),
#                 ha='center', va='center', fontsize=14, color='black', xytext=(0, -model/2*2.5),
#                 textcoords='offset points')
#
#     mae.append(abs(actual - model)*100/actual)
#
# plt.axhline(y=76, color='r', linestyle=':', label='NDWR Crop Inventories 2016')
#
#
# # Set plot labels and title
# plt.xlabel('Year')
# plt.ylabel('Volume (1000s of acre-ft)')
# plt.title('Total Pumping vs Modeled Pumping Diamond Valley')
#
# plt.ylim(0, 100)
#
# # Show the legend
# plt.legend()
#
# plt.savefig(r'with_lower_filter/dv_bar_plot_basin.png', dpi=300)
#
# plt.show()


################################################################
#                   Diamond Valley Prediceted Pumping (Whole Basin) Metric
################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import et data
df = pd.read_csv('../openet_data/dv_openet_data_2018_2022_all.csv')
df['annual_net_et_mm'] = df['annual_et_mm']-df['annual_precip_eff_mm']
df = df.loc[df['annual_net_et_mm'] > 0, :] # filter out areas with less thatn 10 acres
df['est_pumping_mm'] = df['annual_net_et_mm']*1.1142832550149728 # this changed from the original 1.1
df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000

# Get estimated data
# df = df.rename(columns={'year': 'YEAR'})
et_df = df.loc[:, ['YEAR',]].copy()
et_df['data'] = df['est_pumping_m3']
et_df = et_df.groupby('YEAR').sum()
et_df = et_df.reset_index()
et_df['dataset'] = 'Model'

# Get estimated data
pumping_df = et_df.copy()
pumping_df['data'] = [75297823, 68638385.42, 88459789.77, 77087599.33, 78900508.69]
pumping_df['dataset'] = 'Actual'

# Convert to acre-ft
# combine data
df_plot = pd.concat([et_df, pumping_df]).sort_values(by=['YEAR', 'data'])
# df_plot['data'] = df_plot['data']* 0.000810714/1000
df_plot['YEAR'] = df_plot['YEAR'].astype(int)
df_plot = df_plot.rename(columns={'YEAR': 'year'})

# Sample data structure (replace this with your actual data)
data = {
    'Year': df_plot.year,
    'Category': df_plot.dataset,
    'Value': df_plot.data
}

df = pd.DataFrame(data)

# Convert to millions of m3
df['Value'] = df['Value'] / 1e+6

# Create a barplot using Seaborn
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Replace 'Value' and 'Category' with your actual column names
ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')


for p in ax.patches[0:10]:
    ax.annotate(round(p.get_height(), 1), (p.get_x() + p.get_width() / 2., p.get_height()+1),
                ha='center', va='center', fontsize=14, color='black', xytext=(0, 5),
                textcoords='offset points')

mae = []
for index in range(0, 5):
    model = ax.patches[index].get_height()
    actual = ax.patches[index+5].get_height()
    percent =  round((actual - model)*100/actual)
    p = ax.patches[index]
    
    ax.annotate(f'{percent}%', (p.get_x() + p.get_width() / 2., p.get_height()+1),
                ha='center', va='center', fontsize=14, color='black', xytext=(0, -model/2*2.5),
                textcoords='offset points')

    mae.append(abs(actual - model)*100/actual)

plt.axhline(y=76*1.233, color='r', linestyle=':', label='NDWR Crop Inventories 2016')

# Set plot labels and title
plt.xlabel('Year', fontsize=18)
plt.ylabel('GP volume (Mm3)', fontsize=18)

plt.ylim(0, 100*1.233)

# Show the legend
plt.legend(fontsize=14)

plt.savefig(r'with_lower_filter/dv_bar_plot_basin_metric.png', dpi=600)

plt.show()

# MAE 6.911209908661978

# ##############################################################
# #                  Harney Basin Water Use Plot
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/harney_basin/effective_precip_fraction.csv')

# # Sample data for two datasets
# eto = df['ETpot'].values
# niwr = df['NIWR'].values
# eta = df['ETact'].values
# ppt = df['PPT'].values

# # Define the x-axis labels for each year
# years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(eto)), y=eto, color='ForestGreen', label='ASCE Grass Reference ET', linestyle=':')
# sns.scatterplot(x=range(len(eto)), y=eto, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(eta)), y=eta, color='black', label='Modeled Actual ET', linestyle='-.')
# sns.scatterplot(x=range(len(eta)), y=eta, color='black', marker='o')

# sns.lineplot(x=range(len(niwr)), y=niwr, color='SteelBlue', label='Net Irrigation Water Requirement', linestyle='--')
# sns.scatterplot(x=range(len(niwr)), y=niwr, color='SteelBlue', marker='o')

# sns.lineplot(x=range(len(ppt)), y=ppt, color='FireBrick', label='Total Precipitation', linestyle='-')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis

# # Set y limits
# ax1.set_ylim(0, 1250)  # Set the y-axis limits for Dataset 1
# ax1.set_ylim(0, 1250) 

# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.set_ylabel('Flux (mm)')

# # ax1.legend(loc='lower left')
# ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

# plt.title('Harney Basin')

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/hb_et_demands_plot.png', dpi=300)

# # Show the plot
# plt.show()


# ##############################################################
# #                  Harney Basin Precip Plot
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/harney_basin/effective_precip_fraction.csv')

# # Sample data for two datasets
# ppt = df['PPT'].values
# p_rz = df['P_rz'].values
# p_rz_fraction = df['P_rz_fraction'].values

# # Define the x-axis labels for each year
# years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(ppt)), y=ppt, color='ForestGreen', label='Total Precipitation', linestyle=':')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', label='Effective Precipitation', linestyle='--')
# sns.scatterplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', marker='o')

# # Create the secondary axis
# ax2 = ax1.twinx()
# sns.lineplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', ax=ax2, label='Fraction of Effective Precipitation')
# sns.scatterplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis
# ax2.grid(False)  # Remove grid lines on the secondary axis

# # Set y limits
# ax1.set_ylim(0, 425)  # Set the y-axis limits for Dataset 1
# ax2.set_ylim(0.5, 1.01)  # Set the y-axis limits for Dataset 1

# # Set labels and legends for both axes
# ax1.set_ylabel('Flux (mm)')
# ax2.set_ylabel('Fraction')
# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.legend(loc='lower left')
# ax2.legend(loc='lower right')

# plt.title('Harney Basin')


# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/hb_precip_plot.png', dpi=300)


# # Show the plot
# plt.show()

# ##############################################################
# #                  Harney Basin P_eft
# ##############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import data
# df = pd.read_csv('../et-demands/harney_basin/effective_precip_fraction.csv')

# # Sample data for two datasets
# ppt = df['PPT'].values
# p_rz = df['P_eft'].values
# p_rz_fraction = df['P_eft_fraction'].values

# # Define the x-axis labels for each year
# years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']

# # Create the primary axis
# plt.figure(figsize=(8, 4))
# ax1 = sns.lineplot(x=range(len(ppt)), y=ppt, color='ForestGreen', label='Total Precipitation', linestyle=':')
# sns.scatterplot(x=range(len(ppt)), y=ppt, color='ForestGreen', marker='o')

# sns.lineplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', label='P_eft', linestyle='--')
# sns.scatterplot(x=range(len(p_rz)), y=p_rz, color='SteelBlue', marker='o')

# # Create the secondary axis
# ax2 = ax1.twinx()
# sns.lineplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', ax=ax2, label='P_eft_fraction')
# sns.scatterplot(x=range(len(p_rz_fraction)), y=p_rz_fraction, color='FireBrick', marker='o')

# # Remove grid lines
# ax1.grid(False)  # Remove grid lines on the primary axis
# ax2.grid(False)  # Remove grid lines on the secondary axis

# # Set y limits
# ax1.set_ylim(0, 425)  # Set the y-axis limits for Dataset 1
# ax2.set_ylim(0.5, 1.01)  # Set the y-axis limits for Dataset 1

# # Set labels and legends for both axes
# ax1.set_ylabel('Flux (mm)')
# ax2.set_ylabel('Fraction')
# ax1.set_xlabel('Year')
# ax1.set_xticks(range(len(years)))
# ax1.set_xticklabels(years)
# ax1.legend(loc='lower left')
# ax2.legend(loc='lower right')

# plt.title('Harney Basin')


# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/hb_p_eft.png', dpi=300)


# # Show the plot
# plt.show()


# ###############################################################
# #                  Harney Basin Box Plot
# ###############################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# df = df[df["pumping_mm"] > 0]

# # Remove meter change data
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# ##### Build Box Plot #####

# # Reformat dataframe for plotting
# df1 = df.loc[:, ['year', 'annual_net_et_mm']].rename(columns={'annual_net_et_mm': 'data'})
# df1['dataset'] = 'Net ET'

# df2 = df.loc[:, ['year', 'pumping_mm']].rename(columns={'pumping_mm': 'data'})
# df2['dataset'] = 'Pumping'


# df_plot = pd.concat([df1, df2], axis=0)

# # Sample data as Pandas DataFrames (replace this with your data)
# data = pd.DataFrame({
#     'Year': df_plot.year,
#     'Dataset': df_plot.dataset,
#     'Data': df_plot.data
# })

# # Create a publication-quality plot
# sns.set(style="whitegrid", font_scale=1.2)  # Set the plot style and increase font size
# plt.figure(figsize=(10, 6))

# # Define your custom color palette
# colors = {"Net ET": "#4682B4", "Pumping": "#228B22"}

# # Create the box plot
# ax = sns.boxplot(data=data, x='Year', y='Data', hue='Dataset',
#                   width=0.6, palette=colors, dodge=True)

# # Customize plot appearance
# ax.set_xlabel('Year', fontsize=14)
# ax.set_ylabel('Depth (mm)', fontsize=14)
# ax.legend() # (title='Dataset', fontsize=12, title_fontsize=12)

# # Customize tick label font size
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)

# # Add grid lines
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Remove top and right spines
# sns.despine()

# # Set the title and adjust its font size
# ax.set_title('Annual Pumping and Net ET Harney Basin', fontsize=16)

# # Save the plot as a publication-quality image (e.g., PDF, SVG, or high-resolution PNG)
# plt.tight_layout()
# plt.savefig(r'without_lower_filter/hb_box_plot.png', dpi=300)

# # Show the plot
# plt.show()


# ################################################################
# #                   Harney Basin Outlier
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')

# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# # Meter df
# outlier_df = df.loc[(df['pumping_net_et_factor_annual']>1.5) |
#                     # (df['pumping_net_et_factor_annual']<0.5) |
#                     (df["pumping_mm"] == 0) |
#                     (df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])), :]

# data_df = df.loc[(df['pumping_net_et_factor_annual']<1.5) &
#                   # (df['pumping_net_et_factor_annual']>0.5) &
#                   (df["pumping_mm"] > 0) & 
#                   (~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])), :]

# plt.figure(figsize=(8, 8))

# # Set the style to 'whitegrid' or 'ticks' to add gridlines
# sns.set_style('ticks')  # Use 'ticks' for gridlines with ticks

# # Create a scatter plot using DataFrame with different symbols for each dataset
# sns.scatterplot(data=data_df, x='annual_net_et_mm', y='pumping_mm', label='Data', s=50, marker="o")
# sns.scatterplot(data=outlier_df, x='annual_net_et_mm', y='pumping_mm', label='Outlier', s=50, marker="^")

# # Add horizontal line
# plt.axvline(x=640, color='grey', linestyle='--', label='NIWR: 640mm')
# plt.axhline(y=750, color='grey', linestyle='-.', label='NIWR/0.85: 750mm')

# # Adding labels and title with larger fonts
# plt.xlabel("Net ET Flux (mm)", fontsize=18)
# plt.ylabel("Pumping Flux (mm)", fontsize=18)
# plt.title("Scatter Plot of Outliers Harney Basin", fontsize=20)

# # Customize tick label fonts
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Set limit
# plt.ylim(-30, 2400) 
# plt.xlim(0, 800) 

# # Adding a legend with larger fonts
# plt.legend(fontsize=16, loc='upper left')

# # Customize the marker colors
# sns.set_palette("colorblind")  # Use a colorblind-friendly palette

# # Save the plot as a high-resolution image (e.g., PNG)
# plt.savefig(r"without_lower_filter/hb_outlier_scatter_plot.png", dpi=400)

# # Show the plot
# plt.show()


# ################################################################
# #                   Harney Basin Model Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from matplotlib import pyplot as plt

   
# ############## Harney Basin##########################
# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# df = df[df["pumping_mm"] > 0]
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(df.loc[:, data_column])  # Independent variable
# y = np.array(df.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate performance metrics
# r_squared = r2_score(y, X * final_slope)
# mae = mean_absolute_error(y, X * final_slope) * 100 / np.mean(y)
# mse = mean_squared_error(y, X * final_slope)
# rmse = np.sqrt(mse) * 100 / np.mean(y)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
# sns.lineplot(x=new_X,y=final_slope*new_X,label="Linear Regression")
# sns.scatterplot(data=df, x=data_column, y='pumping_mm', label='Meter Data', s=35, marker="o")
# sns.set(style="white")

# # Add text
# plt.text(800, 100, 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}%'.format(final_slope, r_squared, rmse ,mae),
#           fontsize=18, color='black')

# # Add confidence intercal
# plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3, label="95% CI")
# plt.fill_between(new_X, new_X*final_slope+pi_lower, new_X*final_slope+pi_upper, interpolate=True, color='gray', alpha=0.3, label="95% PI")

# plt.legend(fontsize=12, loc='upper left')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET Flux (mm)", fontsize=18)
# plt.ylabel("Pumping Flux (mm)", fontsize=18)
# plt.title("Harney Basin", fontsize=20)

# plt.ylim(-10, 1200)
# plt.xlim(-10, 1200)

# plt.savefig(r"without_lower_filter/hb_model_scatter_plot.png", dpi=400)


# ################################################################
# #                   Harney Basin Model Plot Volume
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from matplotlib import pyplot as plt

   
# ############## Harney Basin##########################
# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# df = df[df["pumping_mm"] > 0]
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# # Calculate volumes
# df['annual_net_et_m3'] = df['annual_net_et_mm']*df['area_m2']/1000/1_000_000 # millions of m3
# df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000/1_000_000 # millions of m3
# data_column = 'annual_net_et_m3'

# # Separeate X and y data
# X = np.array(df.loc[:, data_column])  # Independent variable
# y = np.array(df.loc[:, "pumping_m3"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate performance metrics
# r_squared = r2_score(y, X * final_slope)
# mae = mean_absolute_error(y, X * final_slope) * 100 / np.mean(y)
# mse = mean_squared_error(y, X * final_slope)
# rmse = np.sqrt(mse) * 100 / np.mean(y)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# sns.lineplot(x=[0, new_X.max()], y= [0,new_X.max()], label='1:1 Line')
# sns.lineplot(x=new_X,y=final_slope*new_X,label="Linear Regression")
# sns.scatterplot(data=df, x=data_column, y='pumping_m3', label='Meter Data', s=35, marker="o")
# sns.set(style="white")

# # Add text
# plt.text(0.45, 0.05, 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}%'.format(final_slope, r_squared, rmse ,mae),
#           fontsize=18, color='black')

# # Add confidence intercal
# plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3, label="95% CI")
# plt.fill_between(new_X, new_X*final_slope+pi_lower, new_X*final_slope+pi_upper, interpolate=True, color='gray', alpha=0.3, label="95% PI")

# plt.legend(fontsize=12, loc='upper left')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET Volume (millions of m3)", fontsize=18)
# plt.ylabel("Pumping Volume (millions of m3)", fontsize=18)
# plt.title("Harney Basin", fontsize=20)

# plt.ylim(-0.01, 0.9)
# plt.xlim(-0.01, 0.9)

# plt.savefig(r"without_lower_filter/hb_model_volume_scatter_plot.png", dpi=400)


# ################################################################
# #                   Harney Basin Residule Histogram Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from matplotlib import pyplot as plt

   
# ############## Harney Basin ##########################
# # Import Harney Basin Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# df = df[df["pumping_mm"] > 0]
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(df.loc[:, data_column])  # Independent variable
# y = np.array(df.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
# bin_width = 2 * iqr / (len(residuals) ** (1/3))
# bins=int((max(residuals) - min(residuals)) / bin_width)
# sns.histplot(residuals, bins=bins, kde=False, color='skyblue')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Residuals (mm)", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title("Harney Basin", fontsize=20)

# plt.savefig(r"without_lower_filter/hb_residule_histogram_plot.png", dpi=400)



# ################################################################
# #                   Harney Basin Residule Net ET scatter Plot
# ################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import datetime

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from matplotlib import pyplot as plt

   
# ############## Harney Basin ##########################
# # Import Harney Basin Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
# df = df[df["pumping_mm"] > 0]
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# data_column = 'annual_net_et_mm'

# # Separeate X and y data
# X = np.array(df.loc[:, data_column])  # Independent variable
# y = np.array(df.loc[:, "pumping_mm"]) # Dependent variable 
# # Number of bootstrap samples
# n_samples = 10000

# # Number of data points in the original dataset
# n_data_points = len(X)

# # Confidence level for prediction intervals (e.g., 95%)
# confidence_level = 0.95

# # Create an array to store the bootstrapped coefficients
# bootstrap_coefs = np.zeros((n_samples,))

# # Create lists to store predictions from each bootstrap sample
# bootstrap_predictions = []

# # Perform bootstrap resampling
# for i in range(n_samples):
#     # Randomly sample data points with replacement
#     indices = np.random.choice(n_data_points, n_data_points, replace=True)
#     X_bootstrap = X[indices]
#     y_bootstrap = y[indices]

#     # Fit a linear regression model with intercept=0
#     model = LinearRegression(fit_intercept=False)
#     model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

#     # Store the coefficient
#     bootstrap_coefs[i] = model.coef_[0]

#     # Generate predictions for new data points
#     new_X = np.linspace(0, X.max(), n_data_points)  # New data points
#     new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
#     bootstrap_predictions.append(new_predictions)

# # Calculate confidence intervals for the slope
# final_slope = np.mean(bootstrap_coefs)
# ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
# ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

# # Calculate prediction intervals for new data points
# prediction_y = X*final_slope
# residuals = y - prediction_y
# pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
# pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

# # Build Figure
# plt.figure(figsize=(8, 8))

# # Add data
# # Add data
# sns.scatterplot(x=X, y=residuals, s=35, marker="o")
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Net ET (mm)", fontsize=18)
# plt.ylabel("Residuals (mm)", fontsize=18)
# plt.title("Harney Basin", fontsize=20)

# plt.savefig(r"without_lower_filter/hb_model_residule_scatter_plot.png", dpi=400)


# ################################################################
# #                  Harney Basin Prediceted Pumping
# ################################################################

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Import et data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')
# df['est_pumping_mm'] = df['annual_net_et_mm']*1.15
# df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000
# df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000
# df = df.loc[df['pumping_net_et_factor_annual']<1.5, :]
# # df = df.loc[df['pumping_net_et_factor_annual']>0.5, :]

# # Get estimated data
# et_df = df.loc[:, ['year',]].copy()
# et_df['data'] = df['est_pumping_m3']
# et_df = et_df.groupby('year').sum()
# et_df = et_df.reset_index()
# et_df['dataset'] = 'model'

# # Get estimated data
# pumping_df = df.loc[:, ['year',]].copy()
# pumping_df['data'] = df['pumping_m3']
# pumping_df = pumping_df.groupby('year').sum()
# pumping_df = pumping_df.reset_index()
# pumping_df['dataset'] = 'actual'

# # combine data
# df_plot = pd.concat([et_df, pumping_df]).sort_values(by=['year', 'data'])
# df_plot['year'] = df_plot['year'].astype(int)
# df_plot['data'] = df_plot['data']/1_000_000

# # Sample data structure (replace this with your actual data)
# data = {
#     'Year': df_plot.year,
#     'Category': df_plot.dataset,
#     'Value': df_plot.data
# }

# df = pd.DataFrame(data)

# # Create a barplot using Seaborn
# plt.figure(figsize=(10, 6))
# sns.set_theme(style="whitegrid")

# # Replace 'Value' and 'Category' with your actual column names
# ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')

# for p in ax.patches:
#     ax.annotate(round(p.get_height(), 1), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
#                 textcoords='offset points')

# # Set plot labels and title
# plt.xlabel('Year')
# plt.ylabel('volume (Mm3)')
# plt.title('Total Pumping vs Modeled Pumping Harney Basin')

# plt.ylim(0, 10)

# # Show the legend
# plt.legend()

# plt.savefig(r'without_lower_filter/hb_bar_plot.png', dpi=300)

# plt.show()


# ################################################################
# #                  Harney Basin Outlier Type Figures
# ################################################################

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Import Diamond Valley Data
# df = pd.read_csv(r'../joined_data/hb_joined_et_pumping_data_all.csv')

# # Data dictionary
# data_dict = {
#     'VLM': [df.loc[df.method=='VLM'], '+', 'green'],
#     'ECF': [df.loc[df.method=='ECF'], '^', 'red'],
#     'FMT': [df.loc[df.method=='FMT'], 'o', 'blue'],
#     'OTH': [df.loc[df.method=='OTH'], 'd', 'black'],
#     'PWR': [df.loc[df.method=='PWR'], 's', 'pink'],
#     'NR': [df.loc[df.method=='NR'], '<', 'purple']  
#     }


# # Meter df
# plt.figure(figsize=(8, 8))

# # Set the style to 'whitegrid' or 'ticks' to add gridlines
# sns.set_style('ticks')  # Use 'ticks' for gridlines with ticks

# for type_ in data_dict:
#     id_ = data_dict[type_]

#     # Create a scatter plot using DataFrame with different symbols for each dataset
#     sns.scatterplot(data=id_[0], x='annual_net_et_mm', y='pumping_mm', label=type_, s=50, marker=id_[1], color=id_[2])


# sns.lineplot(x=[0, 2000], y=[0, 2000], label='1:1 line', color='black')

# # Add horizontal line
# plt.axvline(x=640, color='grey', linestyle='--', label='NIWR: 640mm')
# plt.axhline(y=750, color='grey', linestyle='-.', label='NIWR/0.85: 750mm')

# # Adding labels and title with larger fonts
# plt.xlabel("Net ET Depth (mm)", fontsize=18)
# plt.ylabel("GP Depth (mm)", fontsize=18)
# # plt.title("Harney Basin Outliers with Pumping Method", fontsize=20)

# # Customize tick label fonts
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Set limit
# plt.ylim(-30, 2400) 
# plt.xlim(0, 800) 

# # Adding a legend with larger fonts
# plt.legend(fontsize=14, loc='upper left')

# # Customize the marker colors
# sns.set_palette("colorblind")  # Use a colorblind-friendly palette

# # Save the plot as a high-resolution image (e.g., PNG)
# plt.savefig(r"without_lower_filter/hb_outlier_method_scatter_plot.png", dpi=400)

# # Show the plot
# plt.show()




################################################################
#                  Harney Basin Prediceted Pumping (Whole Basin) Metric
################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import et data
df = pd.read_csv('../joined_data/hb_joined_et_pumping_data_all.csv')
df['est_pumping_mm'] = df['annual_net_et_mm']*1.2
df['est_pumping_m3'] = df['est_pumping_mm']*df['area_m2']/1000
df['pumping_m3'] = df['pumping_mm']*df['area_m2']/1000

# Get estimated data
et_df = df.loc[:, ['year',]].copy()
et_df['data'] = df['est_pumping_m3']
et_df = et_df.groupby('year').sum()
et_df = et_df.reset_index()
et_df['dataset'] = 'Model'

# Get estimated data
pumping_df = df.loc[:, ['year',]].copy()
pumping_df['data'] = df['pumping_m3']
pumping_df = pumping_df.groupby('year').sum()
pumping_df = pumping_df.reset_index()
pumping_df['dataset'] = 'Actual'

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
sns.set_theme(style="whitegrid")

# Replace 'Value' and 'Category' with your actual column names
ax = sns.barplot(data=df, x='Year', y='Value', hue='Category')

for p in ax.patches[0:14]:
    ax.annotate(round(p.get_height(), 1), (p.get_x() + p.get_width() / 2., p.get_height()+1),
                ha='center', va='center', fontsize=14, color='black', xytext=(0, -12),
                textcoords='offset points')

mae = []
for index in range(0, 7):
    model = ax.patches[index].get_height()
    actual = ax.patches[index+7].get_height()
    percent =  round((actual - model)*100/actual)
    p = ax.patches[index]
        
    ax.annotate(f'{percent}%', (p.get_x() + p.get_width() / 2., p.get_height()+1),
                ha='center', va='center', fontsize=14, color='black', xytext=(0, -(model/15)*190),
                textcoords='offset points')

    mae.append(abs(actual - model)*100/actual)

# Set plot labels and title
plt.xlabel('Year', fontsize=14)
plt.ylabel('GP Volume (Mm3)', fontsize=14)
# plt.title('Total Pumping vs Modeled Pumping Harney Basin')

plt.ylim(0, 15)

# Show the legend
plt.legend(fontsize=14)

plt.savefig(r'with_lower_filter/HB_bar_plot_basin_metric.png', dpi=600)

plt.show()

# MAE 16.732361947892258
