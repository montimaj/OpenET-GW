import os

import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


def build_outlier_interval_dict(make_plots=False):
    et_vars = {
        'ensemble': 'OpenET Ensemble',
        'ssebop': 'SSEBop',
        'eemetric': 'eeMETRIC',
        'pt_jpl': 'PT-JPL',
        'sims': 'SIMS',
        'geesebal': 'geeSEBAL',
        'disalexi': 'ALEXI/DisALEXI'
    }
    interval_dict = {}
    dv_df = pd.read_csv(f'../machine_learning/dv_joined_ml_pumping_data.csv')
    # dv_df = pd.read_csv(f'../machine_learning/hb_joined_ml_pumping_data.csv')
    # dv_df = dv_df[~dv_df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
    dv_df = dv_df[dv_df.pumping_mm > 0]
    years = range(2018, 2023)
    for et_var, et_name in et_vars.items():
        net_et_factor = f'pumping_net_et_{et_var}_factor_annual'
        ll_dict = {}
        ul_dict = {}
        limit_df = pd.DataFrame()
        for year in years:
            yearly_df = dv_df[dv_df.year == year]
            if et_var == 'ensemble':
                print(year, yearly_df.shape[0])

            q1 = np.percentile(yearly_df[net_et_factor], 25)
            q3 = np.percentile(yearly_df[net_et_factor], 75)
            iqr = q3 - q1
            ll = q1 - 1.5 * iqr
            if ll < 0:
                ll = np.min(yearly_df[net_et_factor])
            ul = q3 + 1.5 * iqr
            ll_dict[year] = np.round(ll, 2)
            ul_dict[year] = np.round(ul, 2)
            l_df = pd.DataFrame(data={
                'Year': [str(year)],
                'Lower limit': [ll_dict[year]],
                'Upper limit': [ul_dict[year]]
            })
            limit_df = pd.concat([limit_df, l_df])
        median_year = np.median(years)
        ll = ll_dict[median_year]
        ul = ul_dict[median_year]
        interval_dict[et_var] = (ll, ul)
        if make_plots:
            plt.figure(figsize=(8, 8))
            sns.boxplot(
                y=dv_df[net_et_factor],
                x='year',
                data=dv_df
            )
            sns.scatterplot(
                x='Year',
                y='Lower limit',
                data=limit_df,
                color='r',
                style='Lower limit',
                markers=['^'],
                s=100
            )
            ul_plot = sns.scatterplot(
                x='Year',
                y='Upper limit',
                data=limit_df,
                color='r',
                style='Upper limit',
                markers=['v'],
                s=100
            )
            # extract the existing handles and labels
            h, l = ul_plot.get_legend_handles_labels()

            # slice the appropriate section of l and h to include in the legend
            ul_plot.legend(
                [h[-1], h[0]],
                ['Upper limit', 'Lower limit'],
                bbox_to_anchor = (0.9, 0.05),
                loc='lower right',
            )
            plt.ylim(-0.3, 3.25)
            plt.xticks(fontsize=16)
            yticks = np.arange(0, 3.1, 0.5).tolist() + [0.7]
            plt.yticks(yticks, fontsize=16)
            plt.xlabel("Year", fontsize=18)
            plt.ylabel("GP / Net ET", fontsize=18)
            plt.axhline(y=1.5, linestyle='--', linewidth=0.5)
            plt.axhline(y=0.7, linestyle='--', linewidth=0.5)

            plt.savefig(f"dv_ratio_boxplot_{et_var}.png", dpi=600, bbox_inches='tight')
            plt.clf()
    print(interval_dict)

    return interval_dict


# def generate_scatter_plots():

# Build Figure
# plt.figure(figsize=(8, 8))
#
# # Add data
# sns.lineplot(x=[0, new_X.max()], y=[0, new_X.max()], label='1:1 Line')
# sns.lineplot(x=new_X, y=final_slope * new_X, label="Linear Regression")
# sns.scatterplot(data=df, x=data_column, y='pumping_mm', label='Meter Data', s=35, marker="o")
# sns.set_style('white', rc={
#     'xtick.bottom': True,
#     'ytick.left': True,
# })
#
# # Add text
# plt.text(1150, 100,
#          'y = {:.2f}*x \n$R^2$ = {:.2f} \nRMSE = {:.2f}% \nMAE = {:.2f}% \nCV = {:.2f}%'.format(
#              final_slope, r_squared,
#              rmse, mae, cv
#          ),
#          fontsize=18,
#          color='black'
# )
#
# # Add confidence intercal
# plt.fill_between(new_X, ci_upper * new_X, ci_lower * new_X, interpolate=True, color='yellow', alpha=0.3,
#                  label="95% CI")
# plt.fill_between(new_X, new_X * final_slope + pi_lower, new_X * final_slope + pi_upper, interpolate=True,
#                  color='gray', alpha=0.3, label="95% PI")
#
# plt.legend(fontsize=12, loc='upper left')
#
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.xlabel(f"{et_name} Net ET depth (mm)", fontsize=18)
# plt.ylabel("GP depth (mm)", fontsize=18)
#
# plt.ylim(-5, 1200)
# plt.xlim(-5, 1200)
# plt.savefig(f"{output_dir}{site}_model_scatter_plot_{net_et_factor}_{et_var}.png", dpi=400, bbox_inches='tight')


def build_linear_regression(site='dv'):
    et_vars = {
        'ensemble': 'OpenET Ensemble',
        'ssebop': 'SSEBop',
        'eemetric': 'eeMETRIC',
        'pt_jpl': 'PT-JPL',
        'sims': 'SIMS',
        'geesebal': 'geeSEBAL',
        'disalexi': 'ALEXI/DisALEXI'
    }
    net_et_factors = et_vars.keys()
    interval_dict = build_outlier_interval_dict()
    final_metrics_df_train = pd.DataFrame()
    final_metrics_df_test = pd.DataFrame()
    test_sizes = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    np.random.seed(1234)
    num_seeds = 10
    seed_values = np.random.choice(range(1000), size=num_seeds, replace=False)
    data_df = pd.read_csv(f'../machine_learning/{site}_joined_ml_pumping_data.csv')
    if site == 'hb':
        data_df = data_df[~data_df.fid.isin([
            '15', '533_1102', '1210_1211',
            '1329', '1539_1549_1550', '1692'
        ])]
    data_df = data_df[data_df.pumping_mm > 0]
    n1 = data_df.shape[0]
    # Number of bootstrap samples
    n_samples = 10000
    for factor in net_et_factors:
        print('Working on', factor, 'results for', site, '...')
        ll, ul = interval_dict[factor]
        metrics_df_train = pd.DataFrame()
        metrics_df_test = pd.DataFrame()
        for et_var, et_name in et_vars.items():
            net_et_factor = f'pumping_net_et_{factor}_factor_annual'
            data_column = f'annual_net_et_{et_var}_mm'
            df = data_df[data_df[net_et_factor] < ul]
            df = df[df[net_et_factor] > ll]
            n2 = df.shape[0]
            samples_removed = round((n1 - n2) * 100 / n1, 2)
            # Separeate X and y data
            X = df[data_column].to_numpy()  # Independent variable
            y = df.pumping_mm.to_numpy()  # Dependent variable
            for test_size in test_sizes:
                split_train_test = test_size > 0
                for seed in seed_values:
                    if split_train_test:
                        x_train, x_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size,
                            random_state=seed
                        )
                    else:
                        x_train = X
                        y_train = y
                        x_test = None
                        y_test = None

                    # Number of data points in the original dataset
                    n_data_points = len(x_train)

                    # Confidence level for prediction intervals (e.g., 95%)
                    # confidence_level = 0.95

                    # Create an array to store the bootstrapped coefficients
                    bootstrap_coefs = np.zeros((n_samples,))

                    # Create lists to store predictions from each bootstrap sample
                    bootstrap_predictions = []
                    np.random.seed(seed)

                    # Perform bootstrap resampling
                    for i in range(n_samples):
                        # Randomly sample data points with replacement
                        indices = np.random.choice(n_data_points, n_data_points, replace=True)
                        X_bootstrap = x_train[indices]
                        y_bootstrap = y_train[indices]

                        # Fit a linear regression model with intercept=0
                        model = LinearRegression(fit_intercept=False)
                        model.fit(X_bootstrap.reshape(-1, 1), y_bootstrap.reshape(-1, 1))

                        # Store the coefficient
                        bootstrap_coefs[i] = model.coef_[0]

                        # Generate predictions for new data points
                        new_X = np.linspace(0, x_train.max(), n_data_points)  # New data points
                        new_predictions = model.predict(new_X.reshape(-1, 1)).flatten()
                        bootstrap_predictions.append(new_predictions)

                    # Calculate confidence intervals for the slope
                    final_slope = np.mean(bootstrap_coefs)
                    # ci_lower = np.percentile(bootstrap_coefs, (1 - confidence_level) * 100 / 2)
                    # ci_upper = np.percentile(bootstrap_coefs, 100 - (1 - confidence_level) * 100 / 2)

                    # Calculate prediction intervals for new data points
                    # prediction_y = x_train * final_slope
                    # residuals = y_train - prediction_y
                    # pi_lower = np.percentile(residuals, (1 - confidence_level) * 100 / 2)
                    # pi_upper = np.percentile(residuals, 100 - (1 - confidence_level) * 100 / 2)

                    # Calculate performance metrics
                    predictions = x_train * final_slope
                    r_squared = r2_score(y_train, predictions)
                    mae = mean_absolute_error(y_train, predictions) * 100 / np.mean(y_train)
                    rmse = mean_squared_error(y_train, predictions, squared=False) * 100 / np.mean(y_train)
                    cv = np.std(predictions) * 100 / np.mean(predictions)

                    et_df_train = pd.DataFrame({
                        'ET Model': [et_name],
                        'Net ET Factor': [factor],
                        'R2': [r_squared],
                        'MAE (%)': [mae],
                        'RMSE (%)': [rmse],
                        'CV (%)': [cv],
                        'Sample Removed (%)': [samples_removed],
                        'Sample size': [n2],
                        'Test Size': [test_size],
                        'Random Seed': [seed],
                        'Slope': [final_slope]
                    })
                    metrics_df_train = pd.concat([metrics_df_train, et_df_train])
                    if split_train_test:
                        predictions = x_test * final_slope
                        r_squared = r2_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions) * 100 / np.mean(y_test)
                        rmse = mean_squared_error(y_test, predictions, squared=False) * 100 / np.mean(y_test)
                        cv = np.std(predictions) * 100 / np.mean(predictions)

                        et_df_test = pd.DataFrame({
                            'ET Model': [et_name],
                            'Net ET Factor': [factor],
                            'R2': [r_squared],
                            'MAE (%)': [mae],
                            'RMSE (%)': [rmse],
                            'CV (%)': [cv],
                            'Sample Removed (%)': [samples_removed],
                            'Sample size': [n2],
                            'Test Size': [test_size],
                            'Random Seed': [seed],
                            'Slope': [final_slope]
                        })
                        metrics_df_test = pd.concat([metrics_df_test, et_df_test])
        final_metrics_df_train = pd.concat([final_metrics_df_train, metrics_df_train])
        final_metrics_df_test = pd.concat([final_metrics_df_test, metrics_df_test])
    data_list = ['TRAIN', 'TEST']
    metrics_df = [final_metrics_df_train, final_metrics_df_test]
    print('Full model stats...')
    metric_df_full = final_metrics_df_train[final_metrics_df_train['Test Size'] == 0]
    metric_df_full.to_csv(f'LM_ET_Comparison_Full_{site}.csv', index=False)
    drop_attrs = [
        'Net ET Factor', 'Sample Removed (%)',
        'Sample size', 'Test Size',
        'Random Seed', 'Slope'
    ]
    metric_df_full = metric_df_full.drop(columns=drop_attrs)
    scores = ['R2', 'MAE (%)', 'RMSE (%)', 'CV (%)']
    mean_metrics = metric_df_full.groupby('ET Model')[scores].mean().round(2).reset_index()
    print(mean_metrics)
    mean_metrics.to_csv(f'Mean_LM_ET_Comparison_Full_{site}.csv', index=False)
    for data, metric_df in zip(data_list, metrics_df):
        metric_df = metric_df[metric_df['Test Size'] > 0]
        metric_df.to_csv(f'LM_ET_Comparison_{data}_{site}.csv', index=False)
        metric_df = metric_df.drop(columns=drop_attrs)
        mean_metrics = metric_df.groupby('ET Model')[scores].mean().round(2).reset_index()
        print('\n', data)
        print(mean_metrics)
        mean_metrics.to_csv(f'Mean_LM_ET_Comparison_{data}_{site}.csv', index=False)


def analysis_plots(site='dv'):
    print(site, '\n')
    scores = ['R2', 'RMSE (%)', 'MAE (%)', 'CV (%)']
    if site == 'hb':
        metric_df = pd.read_csv(f'LM_ET_Comparison_Full_{site}.csv')
        mean_metrics = metric_df.groupby('ET Model')[scores + ['Slope']].mean().round(2).reset_index()
        print(site)
        print(mean_metrics)
        return
    data_list = ['TRAIN', 'TEST']
    test_attrs = [
        'Net ET Factor',
        'Random Seed',
        'Test Size'
    ]
    et_vars = {
        'ensemble': 'OpenET Ensemble',
        'ssebop': 'SSEBop',
        'eemetric': 'eeMETRIC',
        'pt_jpl': 'PT-JPL',
        'sims': 'SIMS',
        'geesebal': 'geeSEBAL',
        'disalexi': 'ALEXI/DisALEXI'
    }


    et_models = et_vars.keys()
    num_et_models = len(et_models)
    for data in data_list:
        metric_df = pd.read_csv(f'LM_ET_Comparison_{data}_{site}.csv')
        if data == 'TRAIN':
            metric_df = pd.concat([metric_df, pd.read_csv(f'LM_ET_Comparison_Full_{site}.csv')])
        mean_metrics = metric_df.groupby('ET Model')[scores + ['Slope']].mean().round(2).reset_index()
        print('\n', data)
        print(mean_metrics)
        # for attr in test_attrs:
        #     mean_metrics = metric_df.groupby(['ET Model', attr])[scores].mean().round(2).reset_index()
        #     analysis_dir = f'Analysis/{attr}/'
        #     if not os.path.exists(analysis_dir):
        #         os.makedirs(analysis_dir)
        #     if attr != 'Net ET Factor':
        #         if attr == 'Test Size':
        #             mean_metrics['Test Size (%)'] = mean_metrics['Test Size'] * 100
        #             mean_metrics['Test Size (%)'] = mean_metrics['Test Size (%)'].astype(int)
        #             if data == 'TRAIN':
        #                 mean_metrics['Train Size (%)'] = 100 - mean_metrics['Test Size (%)']
        #         mean_metrics.to_csv(f'{analysis_dir}Mean_LM_ET_Comparison_{attr}_{data}_{site}.csv', index=False)
        #         for score in scores:
        #             plt.figure(figsize=(25, 10))
        #             plt.rcParams["font.size"] = 16
        #             hue = attr
        #             color = 'crest'
        #             if data == 'TRAIN' and attr == 'Test Size':
        #                 hue = 'Train Size (%)'
        #                 color = 'crest_r'
        #             if attr == 'Random Seed':
        #                 color = 'tab10'
        #             sns.barplot(mean_metrics, y='ET Model', x=score, hue=hue, palette=color)
        #             plt.xlabel(score, fontsize=16)
        #             plt.ylabel('ET Model', fontsize=16)
        #             plt.xticks(fontsize=16)
        #             plt.yticks(fontsize=16)
        #             score_name = score.split(' ')[0]
        #             plt.savefig(
        #                 f"{analysis_dir}ET_Comp_Plots_{attr}_{data}_{score_name}_{site}.png",
        #                 dpi=600, bbox_inches='tight'
        #             )
        #     else:
        #         score_dict = {}
        #         annot_value_dict = {}
        #         for score in scores:
        #             score_dict[score] = np.zeros((num_et_models, num_et_models))
        #             annot_value_dict[score] = np.zeros((num_et_models, num_et_models))
        #             mask = np.zeros_like(score_dict[score], dtype=bool)
        #             mask[np.triu_indices_from(mask)] = True
        #             plt.figure(figsize=(15, 10))
                    # for row, model1 in enumerate(et_models):
                    #     for col, model2 in enumerate(et_models):
                    #         if model1 != model2:
                    #             model1_metric = mean_metrics[(mean_metrics['ET Model'] == et_vars[model1]) & ()][[score]].mean()
                    #             model2_metric = mean_metrics[mean_metrics['ET Model'] == et_vars[model2]][[score]].mean()
                    #             p_value = wilcoxon(model1_metric, model2_metric).pvalue
                    #         reject = p_value < 0.05
                    #         t_df['Model1'] = [model1]
                    #         t_df['Model2'] = [model2]
                    #         t_df['p_value'] = [p_value]
                    #         t_df['Reject'] = [reject]
                    #         ttest_df = pd.concat([ttest_df, t_df])
                    #         p_value_dict[metric][row, col] = p_value
                    #         annot_value_dict[metric][row, col] = reject
                    # ttest_df.to_csv(f'{output_dir}Model_Significance_{metric}.csv', index=False)
                    # ttest_df['Model1'] = ttest_df['Model1'].replace({'ETR': 'ERT', 'LR': 'MLR'})
                    # ttest_df['Model2'] = ttest_df['Model2'].replace({'ETR': 'ERT', 'LR': 'MLR'})
                    # annot = annot_value_dict[metric].astype(bool).astype(str)
                    # model_names = ttest_df['Model1'].unique()
                    # plt.figure()
                    # sns.heatmap(
                    #     p_value_dict[metric],
                    #     mask=mask,
                    #     xticklabels=model_names,
                    #     yticklabels=model_names,
                    #     annot=annot,
                    #     fmt='',
                    #     cmap='crest',
                    #     cbar_kws={'label': r'$p$ value, Wilcoxon signed-rank test'}
                    # )
                    # plt.savefig(
                    #     f'{analysis_dir}ET_HeatMap_{data}_{score}_{site}.png',
                    #     dpi=600, bbox_inches='tight'
                    # )


if __name__ == '__main__':
    # build_linear_regression(site='dv')
    # build_linear_regression(site='hb')
    # analysis_plots('dv')
    # analysis_plots('hb')
    build_outlier_interval_dict(True)