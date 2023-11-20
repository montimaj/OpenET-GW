import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


def build_linear_regression():
    et_vars = {
        'ensemble': 'OpenET Ensemble',
        'ssebop': 'SSEBop',
        'eemetric': 'eeMETRIC',
        'pt_jpl': 'PT-JPL',
        'sims': 'SIMS',
        'geesebal': 'geeSEBAL',
        'disalexi': 'ALEXI/DisALEXI'
    }
    metrics_df = pd.DataFrame()

    for et_var, et_name in et_vars.items():
        net_et_factor = f'pumping_net_et_{et_var}_factor_annual'
        data_column = f'annual_net_et_{et_var}_mm'
        df = pd.read_csv('../machine_learning/dv_joined_ml_pumping_data.csv')
        dv_data = df.loc[df[net_et_factor] < 1.5, :]
        dv_data = dv_data.loc[dv_data[net_et_factor] > 0.5, :]
        dv_data = dv_data[dv_data["pumping_mm"] > 0]

        # Separeate X and y data
        X = np.array(dv_data.loc[:, data_column])  # Independent variable
        y = np.array(dv_data.loc[:, "pumping_mm"])  # Dependent variable
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
        prediction_y = X * final_slope
        residuals = y - prediction_y
        pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
        pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

        # Calculate performance metrics
        r_squared = r2_score(y, X * final_slope)
        mae = mean_absolute_error(y, X * final_slope) * 100 / np.mean(y)
        mse = mean_squared_error(y, X * final_slope)
        rmse = np.sqrt(mse) * 100 / np.mean(y)

        et_df = pd.DataFrame({
            'ET Model': [et_name],
            'R2': [r_squared],
            'MAE (%)': [mae],
            'RMSE (%)': [rmse]
        })
        metrics_df = pd.concat([metrics_df, et_df])

        # Build Figure
        plt.figure(figsize=(8, 8))

        # Add data
        sns.lineplot(x=[0, new_X.max()], y=[0, new_X.max()], label='1:1 Line')
        sns.lineplot(x=new_X, y=final_slope * new_X, label="Linear Regression")
        sns.scatterplot(data=dv_data, x=data_column, y='pumping_mm', label='Meter Data', s=35, marker="o")
        sns.set(style="white")

        # Add text
        plt.text(800, 100,
                 'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}%'.format(final_slope, r_squared, rmse,
                                                                                         mae),
                 fontsize=18, color='black')

        # Add confidence intercal
        plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3,
                         label="95% CI")
        plt.fill_between(new_X, new_X * final_slope + pi_lower, new_X * final_slope + pi_upper, interpolate=True,
                         color='gray', alpha=0.3, label="95% PI")

        plt.legend(fontsize=12, loc='upper left')

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.xlabel(f"{et_name} Net ET depth (mm)", fontsize=18)
        plt.ylabel("GP depth (mm)", fontsize=18)

        plt.ylim(-5, 1200)
        plt.xlim(-5, 1200)

        plt.savefig(f"dv_model_scatter_plot_{et_var}.png", dpi=400)
        plt.clf()

    metrics_df.to_csv('LM_ET_Comparison.csv', index=False)


if __name__ == '__main__':
    build_linear_regression()
    # TODO: HB, Oregon LM ET comparison