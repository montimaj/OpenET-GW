# Compare machine learning models in Diamond Valley, Nevada
# Author: Sayantan Majumdar (sayantan.majumdar@dri.edu)

import os
os.environ['USE_PYGEOS'] = '0'
os.environ["PYTHONWARNINGS"] = "ignore"
import geopandas as gpd
import json
import pandas as pd
import ee
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance


def prepare_ml_data():
    final_csv = 'dv_ml_data_2018_2022.csv'
    if not os.path.exists(final_csv):
        print('Creating ML data...')
        ee.Initialize()
        # General Input Variables
        year_list = [2018, 2019, 2020, 2021, 2022]

        # Effective precip factor
        prz_factor_df = pd.read_csv('../et-demands/diamond_valley/effective_precip_fraction.csv')
        prz_factor_table = prz_factor_df.set_index('Year')['P_rz_fraction'].to_dict()

        # Import shapefile into featureCollection
        gdf = gpd.read_file('../gis/diamond_valley/dv_field_shp/153_DiamondValley_filtered.shp', crs="EPSG:26911")
        gdf = gdf.to_crs("EPSG:4326")
        geo_json = gdf.to_json()
        fc = ee.FeatureCollection(json.loads(geo_json))

        # Import Datasets
        openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
        ssebop_ic = ee.ImageCollection("OpenET/SSEBOP/CONUS/GRIDMET/MONTHLY/v2_0")
        eemetric_ic = ee.ImageCollection("OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0")
        sims_ic = ee.ImageCollection("OpenET/SIMS/CONUS/GRIDMET/MONTHLY/v2_0")
        pt_jpl_ic = ee.ImageCollection("OpenET/PTJPL/CONUS/GRIDMET/MONTHLY/v2_0")
        geesebal_ic = ee.ImageCollection("OpenET/GEESEBAL/CONUS/GRIDMET/MONTHLY/v2_0")
        disalexi_ic = ee.ImageCollection("OpenET/DISALEXI/CONUS/GRIDMET/MONTHLY/v2_0")
        gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        daymet_ic = ee.ImageCollection('NASA/ORNL/DAYMET_V4')
        ndvi_ic = ee.ImageCollection("LANDSAT/LC08/C01/T1_32DAY_NDVI")
        hsg = ee.Image(
            'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/'
            'hydrologic_group'
        )
        soil_depth = ee.Image(
            'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/soil_depth'
        )
        ksat_mean = ee.Image(
            'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/physical/ksat_mean'
        )
        nasa_dem = ee.Image("NASA/NASADEM_HGT/001")

        # Add area to fc
        def add_area(fc):
            area = fc.area()
            return fc.set({'area_m2': area})

        fc = fc.map(add_area)
        df_list = []
        static_img_ic = [hsg, soil_depth, ksat_mean, nasa_dem]
        static_df_list = []
        for idx, static_img in enumerate(static_img_ic):
            reducer = ee.Reducer.mean()
            stat = 'mean'
            if idx == 0:
                reducer = ee.Reducer.mode()
                stat = 'mode'
            elif idx == 3:
                stat = 'elevation'
            static_fc = static_img.reduceRegions(
                collection=fc,
                reducer=reducer,
                scale=30,
                tileScale=1
            )
            url = static_fc.getDownloadURL('csv')
            df = pd.read_csv(url)
            static_df_list.append(df[stat])

        for year in year_list:
            print(year)

            # Precip factor
            prz_factor = prz_factor_table[year]

            # Calendar year based
            # Build OpenET image
            openet_ensemble = openet_ic.select('et_ensemble_mad') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_ensemble_mm')
            ssebop_et = ssebop_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_ssebop_mm')
            eemetric_et = eemetric_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_eemetric_mm')
            sims_et = sims_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_sims_mm')
            pt_jpl_et = pt_jpl_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_pt_jpl_mm')
            geesebal_et = geesebal_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_geesebal_mm')
            disalexi_et = disalexi_ic.select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_et_disalexi_mm')

            # Build gridMET image
            gridmet_precip = gridmet_ic.select('pr') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_gridmet_precip_mm')

            gridmet_precip_eff = gridmet_precip.multiply(prz_factor).rename('annual_gridmet_precip_eff_mm')

            gridmet_tmmx = gridmet_ic.select('tmmx') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median() \
                .rename('annual_tmmx_K')
            gridmet_tmmn = gridmet_ic.select('tmmn') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median() \
                .rename('annual_tmmn_K')
            gridmet_eto = gridmet_ic.select('eto') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_eto_mm')
            gridmet_etr = gridmet_ic.select('etr') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_etr_mm')
            gridmet_vpd = gridmet_ic.select('vpd') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_vpd_kPa')
            gridmet_vs = gridmet_ic.select('vs') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .mean() \
                .rename('annual_vs_mps')
            gridmet_rmax = gridmet_ic.select('rmax') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median() \
                .rename('annual_rmax')
            gridmet_rmin = gridmet_ic.select('rmin') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median() \
                .rename('annual_rmin')
            daymet_precip = daymet_ic.select('prcp') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum() \
                .rename('annual_dayment_precip_mm')

            daymet_precip_eff = daymet_precip.multiply(prz_factor).rename('annual_daymet_precip_eff_mm')
            ndvi = ndvi_ic.select('NDVI') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .max() \
                .rename('annual_ndvi')
            data_bands = [
                ssebop_et,
                eemetric_et,
                sims_et,
                pt_jpl_et,
                geesebal_et,
                disalexi_et,
                gridmet_precip,
                gridmet_precip_eff,
                gridmet_tmmx,
                gridmet_tmmn,
                gridmet_eto,
                gridmet_etr,
                gridmet_vpd,
                gridmet_vs,
                gridmet_rmax,
                gridmet_rmin,
                daymet_precip,
                daymet_precip_eff,
                ndvi
            ]
            data_band_names = [
                'dri_field_',
                'area_m2',
                'annual_et_ensemble_mm',
                'annual_et_ssebop_mm',
                'annual_et_eemetric_mm',
                'annual_et_sims_mm',
                'annual_et_pt_jpl_mm',
                'annual_et_geesebal_mm',
                'annual_et_disalexi_mm',
                'annual_gridmet_precip_mm',
                'annual_gridmet_precip_eff_mm',
                'annual_tmmx_K',
                'annual_tmmn_K',
                'annual_eto_mm',
                'annual_etr_mm',
                'annual_vpd_kPa',
                'annual_vs_mps',
                'annual_rmax',
                'annual_rmin',
                'annual_daymet_precip_mm',
                'annual_daymet_precip_eff_mm',
                'annual_ndvi'
            ]

            data_img = openet_ensemble
            for band in data_bands:
                data_img = data_img.addBands(band)

            # Filter Over Feature Collection
            data_fc = data_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=30,
                tileScale=1)

            # Download table
            url = data_fc.getDownloadURL('csv', data_band_names)
            df = pd.read_csv(url)
            df['YEAR'] = year
            df['eff_factor'] = prz_factor

            # Append to df
            df_list.append(df)

        final_df = pd.concat(df_list)
        final_df['HSG'] = static_df_list[0]
        final_df['soil_depth_mm'] = static_df_list[1] * 10
        final_df['ksat_mean_micromps'] = static_df_list[2]
        final_df['elevation_m'] = static_df_list[3]
        final_df.to_csv(final_csv, index=False)
    return pd.read_csv(final_csv)


def process_dv_data(ml_data_df):
    year_list = [2018, 2019, 2020, 2021, 2022]
    join_table = pd.read_csv('../joined_data/dv_field_pou_id_join.csv')

    # Read ET and Pumping tables
    # ET table
    et_df = ml_data_df
    pumping_df = pd.read_csv(
        '../pumping_data/diamond_valley/Diamond Valley Data for Meters Database Pumpage Report.csv')

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
        et_df[f'annual_et_{et_var}_m3'] = et_df[f'annual_et_{et_var}_mm'] * et_df['area_m2'] / 1000
    et_df['annual_gridmet_precip_eff_m3'] = et_df['annual_gridmet_precip_eff_mm'] * et_df['area_m2'] / 1000

    et_df['dri_field_'] = et_df['dri_field_'].astype(str)
    et_df['count_et'] = 1

    # Process Pumping data
    pumping_df = pumping_df.loc[pumping_df['Method'] == 'METER', :]  # Filter to only metered data
    pumping_df = pumping_df.loc[:, ['App', 'UsageYear', 'UsageAmountAcreFT', 'MeterDeactivated']]
    pumping_df['pumping_m3'] = pumping_df['UsageAmountAcreFT'] * 1233.48  # acre-ft to m3
    pumping_df['count_pump'] = 1

    # Process each paring
    dict_list = []
    for year in year_list:
        for row in join_table.iterrows():

            # Convert row to dictionary
            row = dict(row[1])

            # Build id dictionar
            id_ = {'year': year,
                   'dri_field_id': row['dri_field_id'],
                   'all_app': row['all_app']}

            # Extract both lists
            dri_id_list = row['dri_field_id'].split('_')
            app_list = row['all_app'].split('_')

            # Extract ET data
            et_df_sub = et_df.loc[(et_df['dri_field_'].isin(dri_id_list)) & (et_df['YEAR'] == year)]
            et_dict = dict(et_df_sub.sum())

            # Extract all app values
            pumping_df_sub = pumping_df.loc[(pumping_df['App'].isin(app_list)) & (pumping_df['UsageYear'] == year)]
            # Check if meter was changed this year (remove if it was)
            meter_change = pumping_df_sub['MeterDeactivated'].sum()
            if meter_change > 0:
                meter_change = 1
            else:
                meter_change = 0

            meter_dict = {'meter_change': meter_change}

            # Drop where pumping data is 0
            count_pump = pumping_df_sub['count_pump'].sum()
            if count_pump == 0:
                continue
            else:
                pass

            pumping_df_sub = pumping_df_sub.loc[:, ['pumping_m3', 'count_pump']]
            pumping_df_sub = pumping_df_sub.drop_duplicates()  # grouped apps have the same pumping data
            pumping_dict = dict(pumping_df_sub.sum())

            # merge dictionaries
            et_dict.update(pumping_dict)
            et_dict.update(id_)
            et_dict.update(meter_dict)

            dict_list.append(et_dict)

    final_df = pd.DataFrame(dict_list)

    # Build addictional units
    final_df['pumping_mm'] = final_df['pumping_m3'] / final_df['area_m2'] * 1000
    for et_var in et_vars.keys():
        final_df[f'annual_net_et_{et_var}_mm'] = (final_df[f'annual_et_{et_var}_m3'] - final_df['annual_gridmet_precip_eff_m3'])/final_df['area_m2'] * 1000
        final_df[f'pumping_net_et_{et_var}_factor_annual'] = np.round(
            final_df['pumping_mm'] / final_df[f'annual_net_et_{et_var}_mm'], 1
        )
    final_df.to_csv('dv_joined_ml_pumping_data.csv', index=False)
    return final_df


def get_model_param_dict(random_state=0):
    model_dict = {
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1
        ),
        'RF': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=-1, bootstrap=True)
    }

    param_dict = {'LGBM': {
        'n_estimators': [300, 400, 500],
        'max_depth': [16, 20, -1],
        'learning_rate': [0.01, 0.05],
        'subsample': [1, 0.9],
        'colsample_bytree': [1, 0.9, 0.8],
        'colsample_bynode': [1, 0.9, 0.8],
        'path_smooth': [0, 0.1, 0.2],
        'num_leaves': [31, 32],
        'min_child_samples': [30, 40, 10]
    }, 'RF': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7, 10, 12, 20, 30, None],
        'max_depth': [8, 15, 20, 6, 10, None],
        'max_leaf_nodes': [16, 20],
        'min_samples_leaf': [1, 2]
    }, 'ETR': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7, 10, 12, 20, 30, None],
        'max_depth': [8, 15, 20, 6, 10, None],
        'min_samples_leaf': [1, 2]
    }}
    return model_dict, param_dict


def get_grid_search_stats(gs_model):
    scores = gs_model.cv_results_
    print('Train Results...')
    r2 = scores['mean_train_r2'].mean()
    rmse = -scores['mean_train_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_train_neg_mean_absolute_error'].mean()
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('Validation Results...')
    r2 = scores['mean_test_r2'].mean()
    rmse = -scores['mean_test_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_test_neg_mean_absolute_error'].mean()
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_prediction_stats(actual_values, pred_values, precision=2):
    r2, mae, rmse = (np.nan,) * 3
    mean_actual = np.mean(actual_values)
    if actual_values.size and pred_values.size:
        r2 = np.round(r2_score(actual_values, pred_values), precision)
        mae = np.round(mean_absolute_error(actual_values, pred_values) * 100 / mean_actual, precision)
        rmse = np.round(mean_squared_error(actual_values, pred_values, squared=False) * 100 / mean_actual, precision)
    return r2, mae, rmse


def build_ml_model(ml_df):
    random_state = 1234
    drop_attrs = [
        'pumping_mm',
        'dri_field_id',
        'dri_field_',
        'area_m2',
        'annual_et_ensemble_m3',
        'annual_gridmet_precip_eff_m3',
        'count_et',
        'pumping_m3',
        'count_pump',
        'year',
        'YEAR',
        'all_app',
        'meter_change'
    ]
    # Uncomment and comment out accordingly to check individual OpenET model performance vs the OpenET ensemble
    drop_attrs_et = [
        'pumping_net_et_ensemble_factor_annual',
        'pumping_net_et_ssebop_factor_annual',
        'pumping_net_et_eemetric_factor_annual',
        'pumping_net_et_pt_jpl_factor_annual',
        'pumping_net_et_sims_factor_annual',
        'pumping_net_et_geesebal_factor_annual',
        'pumping_net_et_disalexi_factor_annual',
        #'annual_net_et_ensemble_mm',
        'annual_net_et_ssebop_mm',
        'annual_net_et_eemetric_mm',
        'annual_net_et_pt_jpl_mm',
        'annual_net_et_sims_mm',
        'annual_net_et_geesebal_mm',
        'annual_net_et_disalexi_mm'
    ]

    # Uncomment and comment out accordingly to select the correct outlier removal factor
    net_et_factor = 'pumping_net_et_ensemble_factor_annual'
    # net_et_factor = 'pumping_net_et_ssebop_factor_annual'
    # net_et_factor = 'pumping_net_et_eemetric_factor_annual'
    #net_et_factor = 'pumping_net_et_pt_jpl_factor_annual'
    # net_et_factor = 'pumping_net_et_sims_factor_annual'
    # net_et_factor = 'pumping_net_et_geesebal_factor_annual'
    # net_et_factor = 'pumping_net_et_disalexi_factor_annual'

    drop_attrs += drop_attrs_et
    model_dict, param_dict = get_model_param_dict(random_state)
    ml_df = ml_df.loc[ml_df[net_et_factor] < 1.5, :]
    ml_df = ml_df.loc[ml_df[net_et_factor] > 0.5, :]
    ml_df = ml_df[ml_df["pumping_mm"] > 0]
    dv_data = pd.get_dummies(ml_df, columns=['HSG'])
    y = dv_data['pumping_mm']
    X = dv_data.drop(columns=drop_attrs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    y = ml_df['pumping_mm']
    scoring_metrics = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
    for model_name in model_dict.keys():
        print('\nSearching best params for {}...'.format(model_name))
        model = model_dict[model_name]
        model_grid = RandomizedSearchCV(
            estimator=model, param_distributions=param_dict[model_name],
            scoring=scoring_metrics, n_jobs=-1, cv=5, refit=scoring_metrics[1],
            return_train_score=True, random_state=random_state
        )
        model_grid.fit(X_train, y_train)
        get_grid_search_stats(model_grid)
        model = model_grid.best_estimator_
        print('Best params: ', model_grid.best_params_)
        imp_dict = {'Features': list(X_train.columns)}
        f_imp = np.array(model.feature_importances_).astype(float)
        if model_name == 'LGBM':
            f_imp /= np.sum(f_imp)
        imp_dict['F_IMP'] = np.round(f_imp, 5)
        imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
        print(imp_df)
        imp_df.to_csv('F_IMP.csv', index=False)
        y_pred_train = np.abs(model.predict(X_train))
        print('Training+Validation metrics...')
        r2, mae, rmse = get_prediction_stats(y_train, y_pred_train)
        print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
        y_pred_test = np.abs(model.predict(X_test))
        print('Test metrics...')
        r2, mae, rmse = get_prediction_stats(y_test, y_pred_test)
        print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
        train_result = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=scoring_metrics[1]
        )
        test_results = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=scoring_metrics[1]
        )
        sorted_importances_idx = train_result.importances_mean.argsort()
        train_importances = pd.DataFrame(
            train_result.importances[sorted_importances_idx].T,
            columns=X.columns[sorted_importances_idx],
        )
        test_importances = pd.DataFrame(
            test_results.importances[sorted_importances_idx].T,
            columns=X.columns[sorted_importances_idx],
        )
        train_importances = train_importances[train_importances.columns[-5:]]
        test_importances = test_importances[test_importances.columns[-5:]]
        for name, importances in zip(["train", "test"], [train_importances, test_importances]):
            plt.figure(figsize=(10, 6))
            plt.rcParams.update({'font.size': 12})
            ax = importances.plot.box(vert=False, whis=10)
            ax.set_xlabel("Decrease in RMSE (mm)")
            ax.axvline(x=0, color="k", linestyle="--")
            ax.figure.tight_layout()
            plt.savefig(f'{model_name}_{name}_PI.png', dpi=400)
            plt.clf()


if __name__ == '__main__':
    ml_data_df = prepare_ml_data()
    ml_data_df = process_dv_data(ml_data_df)
    build_ml_model(ml_data_df)
