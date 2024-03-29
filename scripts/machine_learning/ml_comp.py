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
import scipy as scp
import swifter
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.inspection import permutation_importance


def prepare_ml_data(site='dv'):
    final_csv = f'{site}_ml_data_2018_2022.csv'
    if not os.path.exists(final_csv):
        file_dict = {
            'dv_prz_factor': '../et-demands/diamond_valley/effective_precip_fraction.csv',
            'dv_field_shp': '../gis/diamond_valley/dv_field_shp/153_DiamondValley_filtered.shp',
            'dv_crs': 'EPSG:26911',
            'dv_year_list': range(2018, 2023),
            'dv_field_id': 'dri_field_',
            'hb_prz_factor': '../et-demands/harney_basin/effective_precip_fraction.csv',
            'hb_field_shp': '../gis/harney_basin/harney_fields_2016_gw_well_wu_report_Merge.shp',
            'hb_crs': 'EPSG:26911',
            'hb_year_list': range(2016, 2023),
            'hb_field_id': 'Field_ID',
        }
        print('Creating ML data...')
        ee.Initialize()
        # General Input Variables
        year_list = file_dict[f'{site}_year_list']

        # Effective precip factor
        prz_factor_df = pd.read_csv(file_dict[f'{site}_prz_factor'])
        prz_factor_table = prz_factor_df.set_index('Year')['P_rz_fraction'].to_dict()

        # Import shapefile into featureCollection
        gdf = gpd.read_file(
            file_dict[f'{site}_field_shp'],
            crs=file_dict[f'{site}_crs']
        )
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
                file_dict[f'{site}_field_id'],
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


def correct_hsg(unique_hsgs, hsg):
    if hsg in unique_hsgs:
        return hsg
    hsg_list = []
    for val in hsg:
        hsg_list.append(int(val))
    return str(scp.stats.mode(hsg_list).mode)

def process_hb_data(ml_data_df):
    year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

    # Read ET and Pumping tables
    # ET table
    ml_data_df['HSG'] = ml_data_df['HSG'].astype(int).astype(str)
    et_df = ml_data_df
    pumping_df = pd.read_excel(
        '../pumping_data/harney_basin/Harney_PumpingWell_FieldID_WUR_Relationship_2016_2022.xlsx', header=1)

    # Build join table
    join_table = pumping_df.loc[:, ['WUR_Report_ID', 'FID_1', 'FID_2', 'FID_3']]
    join_table = join_table.dropna(subset=['WUR_Report_ID'])

    join_table_list = []
    for row in join_table.iterrows():
        pass
        row = dict(row[1].dropna().astype(int))
        id_count = len(row.keys())

        if id_count == 2:
            id_str = f"{row['FID_1']}"

        elif id_count == 3:
            id_str = f"{row['FID_1']}_{row['FID_2']}"

        if id_count == 4:
            id_str = f"{row['FID_1']}_{row['FID_2']}_{row['FID_3']}"

        join_table_list.append({'WUR_Report_ID': row['WUR_Report_ID'], 'fid': id_str})

    join_table = pd.DataFrame(join_table_list)

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

    # Process each paring
    dict_list = []
    for year in year_list:
        for row in join_table.iterrows():
            # Convert row to dictionary
            row = dict(row[1])

            # Build id dictionar
            id_ = {'year': year,
                   'WUR_Report_ID': row['WUR_Report_ID'],
                   'fid': row['fid']}

            # Extract both lists
            fid_list = row['fid'].split('_')
            fid_list = [int(x) for x in fid_list]
            app_id = row['WUR_Report_ID']

            # Extract ET data
            et_df_sub = et_df.loc[(et_df['Field_ID'].isin(fid_list)) & (et_df['YEAR'] == year)]
            et_dict = dict(et_df_sub.sum())

            # Extract all app values
            pumping_df_sub = pumping_df.loc[pumping_df['WUR_Report_ID'] == app_id, [
                f'TWU_{year}']] * 1233.48  # convert acre-ft to m3
            pumping_df_sub = pumping_df_sub.rename(columns={f'TWU_{year}': 'pumping_m3'})
            pumping_dict = dict(pumping_df_sub.sum())

            # Pumping outlier
            outlier_df_sub = pumping_df.loc[
                pumping_df['WUR_Report_ID'] == app_id, [f'Outlier_{year}']]  # convert acre-ft to m3
            outlier_df_sub = outlier_df_sub.rename(columns={f'Outlier_{year}': 'outlier'})
            outlier_dict = dict(outlier_df_sub.sum())

            # Check if meter was changed this year (remove if it was)
            method_df = pumping_df.loc[pumping_df['WUR_Report_ID'] == app_id, [f'Meth_{year}']]
            method_df = method_df.rename(columns={f'Meth_{year}': 'method'})
            method_dict = dict(method_df.sum())

            # merge dictionaries
            et_dict.update(pumping_dict)
            et_dict.update(method_dict)
            et_dict.update(outlier_dict)
            et_dict.update(id_)
            dict_list.append(et_dict)

    final_df = pd.DataFrame(dict_list)
    unique_hsgs = ml_data_df.HSG.unique()
    final_df['num_fields'] = final_df.HSG.apply(lambda x: len(x))
    final_df['HSG'] = final_df['HSG'].apply(lambda x: correct_hsg(unique_hsgs, x))

    # Build addictional units
    final_df['pumping_mm'] = final_df['pumping_m3'] / final_df['area_m2'] * 1000
    for et_var in et_vars.keys():
        final_df[f'annual_net_et_{et_var}_mm'] = (final_df[f'annual_et_{et_var}_m3'] - final_df[
            'annual_gridmet_precip_eff_m3']) / final_df['area_m2'] * 1000
        final_df[f'pumping_net_et_{et_var}_factor_annual'] = np.round(
            final_df['pumping_mm'] / final_df[f'annual_net_et_{et_var}_mm'], 1
        )
    pred_cols = [
        'annual_ndvi',
        'ksat_mean_micromps',
        'annual_rmin',
        'soil_depth_mm',
        'annual_rmax',
        'annual_tmmx_K',
        'annual_vs_mps',
        'elevation_m',
        'annual_tmmn_K'
    ]
    for pred_col in pred_cols:
        final_df[pred_col] /= final_df['num_fields']
    final_df.to_csv('hb_joined_ml_pumping_data.csv', index=False)
    return final_df


def process_dv_data(ml_data_df):
    final_csv = 'dv_joined_ml_pumping_data.csv'
    if os.path.exists(final_csv):
        return pd.read_csv(final_csv)
    year_list = [2018, 2019, 2020, 2021, 2022]
    join_table = pd.read_csv('../joined_data/dv_field_pou_id_join.csv')

    # Read ET and Pumping tables
    # ET table
    ml_data_df['HSG'] = ml_data_df['HSG'].astype(int).astype(str)
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
    unique_hsgs = ml_data_df.HSG.unique()
    final_df['num_fields'] = final_df.HSG.apply(lambda x: len(x))
    final_df['HSG'] = final_df['HSG'].apply(lambda x: correct_hsg(unique_hsgs, x))

    # Build addictional units
    final_df['pumping_mm'] = final_df['pumping_m3'] / final_df['area_m2'] * 1000
    for et_var in et_vars.keys():
        final_df[f'annual_net_et_{et_var}_mm'] = (final_df[f'annual_et_{et_var}_m3'] - final_df['annual_gridmet_precip_eff_m3'])/final_df['area_m2'] * 1000
        final_df[f'pumping_net_et_{et_var}_factor_annual'] = np.round(
            final_df['pumping_mm'] / final_df[f'annual_net_et_{et_var}_mm'], 1
        )
    pred_cols = [
        'annual_ndvi',
        'ksat_mean_micromps',
        'annual_rmin',
        'soil_depth_mm',
        'annual_rmax',
        'annual_tmmx_K',
        'annual_vs_mps',
        'elevation_m',
        'annual_tmmn_K'
    ]
    for pred_col in pred_cols:
        final_df[pred_col] /= final_df['num_fields']
    final_df.to_csv(final_csv, index=False)
    return final_df


def neg_root_mean_squared_error_percent(actual_values, pred_values):
    return mean_squared_error(actual_values, pred_values, squared=False) * 100 / np.mean(actual_values)


def neg_mean_absolute_error_percent(actual_values, pred_values):
    return mean_absolute_error(actual_values, pred_values) * 100 / np.mean(actual_values)


def coef_var(actual_values, pred_values):
    return np.std(pred_values) * 100 / np.mean(pred_values)


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
        'n_estimators': [300, 400, 500, 800],
        'max_depth': [8, 15, 20, 6, 10, -1],
        'learning_rate': [0.01, 0.005, 0.05, 0.1],
        'subsample': [1, 0.9, 0.8],
        'colsample_bytree': [1, 0.9],
        'colsample_bynode': [1, 0.9],
        'path_smooth': [0, 0.1, 0.2],
        'num_leaves': [16, 20, 31, 32, 63, 127, 15, 255, 7],
        'min_child_samples': [30, 40, 10, 20],
    }, 'RF': {
        'n_estimators': [300, 400, 500, 800],
        'max_features': [5, 6, 7, 10, 12, 20, 30, None],
        'max_depth': [8, 15, 20, 6, 10, None],
        'max_leaf_nodes': [16, 20, 31, 32, 63, 127, 15, 255, 7, None],
        'min_samples_leaf': [1, 2],
        'max_samples': [None, 0.9],
        'min_samples_split': [2, 3, 4, 0.01]
    }, 'ETR': {
        'n_estimators': [300, 400, 500, 800],
        'max_features': [5, 6, 7, 10, 12, 20, 30, None],
        'max_depth': [8, 15, 20, 6, 10, None],
        'min_samples_leaf': [1, 2],
        'max_samples': [None, 0.9],
        'max_leaf_nodes': [16, 20, 31, 32, 63, 127, 15, 255, 7, None],
        'min_samples_split': [2, 3, 4, 0.01]
    }}
    return model_dict, param_dict


def get_grid_search_stats(gs_model, precision=2):
    scores = gs_model.cv_results_
    print('Train Results...')
    r2 = np.round(scores['mean_train_r2'].mean(), 3)
    rmse = np.round(-scores['mean_train_neg_root_mean_squared_error_percent'].mean(), precision)
    mae = np.round(-scores['mean_train_neg_mean_absolute_error_percent'].mean(), precision)
    cv = np.round(-scores['mean_train_coef_var'].mean(), precision)
    print(f'R2: {r2}, RMSE: {rmse}%, MAE: {mae}%, CV: {cv}%')
    print('Validation Results...')
    r2 = np.round(scores['mean_test_r2'].mean(), 3)
    rmse = np.round(-scores['mean_test_neg_root_mean_squared_error_percent'].mean(), precision)
    mae = np.round(-scores['mean_test_neg_mean_absolute_error_percent'].mean(), precision)
    cv = np.round(-scores['mean_test_coef_var'].mean(), precision)
    print(f'R2: {r2}, RMSE: {rmse}%, MAE: {mae}%, CV: {cv}%')


def get_prediction_stats(actual_values, pred_values, precision=2):
    r2, mae, rmse, cv = (np.nan,) * 4
    mean_actual = np.mean(actual_values)
    if actual_values.size and pred_values.size:
        r2 = np.round(r2_score(actual_values, pred_values), 3)
        mae = np.round(mean_absolute_error(actual_values, pred_values) * 100 / mean_actual, precision)
        rmse = np.round(mean_squared_error(actual_values, pred_values, squared=False) * 100 / mean_actual, precision)
        cv = np.round(np.std(pred_values) / np.mean(pred_values) * 100, precision)
    return r2, mae, rmse, cv


def get_sklearn_field_metrics(actual_value, row, estimator_list, metric_name='CV'):
    pred_arr = np.array([estimator.predict(row) for estimator in estimator_list])
    actual_arr = [actual_value] * pred_arr.size
    if metric_name == 'CV':
        metric = np.std(pred_arr) * 100 / np.mean(pred_arr)
    elif metric_name == 'RMSE':
        metric = mean_squared_error(actual_arr, pred_arr, squared=False) * 100 / actual_value
    else:
        metric = mean_absolute_error(actual_arr, pred_arr) * 100 / actual_value
    return metric


def calc_coeff_var(input_df, drop_attrs, ml_model):
    X = input_df.drop(columns=drop_attrs)
    y = input_df['pumping_mm']
    metric_df = input_df.copy(deep=True)
    if isinstance(ml_model, LGBMRegressor):
        pred_leaf_idx = ml_model.predict(X, pred_leaf=True)
        df = pd.DataFrame({
            "LI": pred_leaf_idx.reshape(-1),
            "x": np.tile(np.arange(pred_leaf_idx.shape[1]), pred_leaf_idx.shape[0]),
            "y": np.repeat(np.arange(pred_leaf_idx.shape[0]), pred_leaf_idx.shape[1])
        })
        df['LV'] = df.swifter.apply(
            lambda row: ml_model.booster_.get_leaf_output(
                tree_id=row.x,
                leaf_id=row.LI
            ),
            axis=1
        )
        df = df[['LV', 'y']].groupby('y')
        coeff_var = df.std() / df.mean()
        metric_df['CV'] = coeff_var
    else:
        metric_df['pred_pumping_mm'] = ml_model.predict(X)
        X['y'] = y
        metrics = ['CV', 'RMSE', 'MAE']
        for metric in metrics:
            metric_df[metric] = X.swifter.apply(
                lambda row: get_sklearn_field_metrics(
                    row[-1],
                    row[:-1].to_numpy().reshape(1, -1),
                    ml_model.estimators_,
                    metric
                ),
                axis=1
            )
    return metric_df


def create_cv_files(input_df, drop_attrs, ml_model):
    df = calc_coeff_var(input_df.copy(deep=True), drop_attrs, ml_model)
    df.to_csv('ML_uncertainty.csv', index=False)


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
    return interval_dict



def build_ml_model(ml_df, site='dv'):
    random_state = 1234
    drop_attr_dict = {
        'dv': [
            'dri_field_id',
            'dri_field_',
            'count_et',
            'all_app',
            'count_pump',
            'meter_change',
        ],'hb': [
            'Field_ID',
            'fid',
            'WUR_Report_ID',
            'outlier',
            'method'
        ]
    }
    drop_attrs = [
        'pumping_mm',
        'area_m2',
        'annual_et_ensemble_m3',
        'annual_gridmet_precip_eff_m3',
        'pumping_m3',
        'year',
        'YEAR',
        'annual_et_ensemble_m3',
        'annual_et_ssebop_m3',
        'annual_et_eemetric_m3',
        'annual_et_pt_jpl_m3',
        'annual_et_sims_m3',
        'annual_et_geesebal_m3',
        'annual_et_disalexi_m3',
        'num_fields'
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
        'annual_net_et_ensemble_mm',
        'annual_net_et_ssebop_mm',
        'annual_net_et_eemetric_mm',
        'annual_net_et_pt_jpl_mm',
        'annual_net_et_sims_mm',
        'annual_net_et_geesebal_mm',
        #'annual_net_et_disalexi_mm',
        'annual_et_ensemble_mm',
        'annual_et_ssebop_mm',
        'annual_et_eemetric_mm',
        'annual_et_pt_jpl_mm',
        'annual_et_sims_mm',
        'annual_et_geesebal_mm',
        #'annual_et_disalexi_mm'
    ]

    # Uncomment and comment out accordingly to select the correct outlier removal factor
    net_et_factor = 'pumping_net_et_ensemble_factor_annual'
    # net_et_factor = 'pumping_net_et_ssebop_factor_annual'
    # net_et_factor = 'pumping_net_et_eemetric_factor_annual'
    # net_et_factor = 'pumping_net_et_pt_jpl_factor_annual'
    # net_et_factor = 'pumping_net_et_sims_factor_annual'
    # net_et_factor = 'pumping_net_et_geesebal_factor_annual'
    # net_et_factor = 'pumping_net_et_disalexi_factor_annual'

    drop_attrs += drop_attrs_et + drop_attr_dict[site]
    model_dict, param_dict = get_model_param_dict(random_state)
    if site == 'hb':
        ml_df = ml_df[~ml_df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
    interval_dict = build_outlier_interval_dict()
    # Uncomment and comment out accordingly to select the correct outlier removal factor
    ll, ul = interval_dict['ensemble']
    # ll, ul = interval_dict['ssebop']
    # ll, ul = interval_dict['eemetric']
    # ll, ul = interval_dict['pt_jpl']
    # ll, ul = interval_dict['sims']
    # ll, ul = interval_dict['geesebal']
    # ll, ul = interval_dict['disalexi']

    print(ll, ul)
    ml_df = ml_df[ml_df[net_et_factor] < ul]
    ml_df = ml_df[ml_df[net_et_factor] > ll]
    ml_df = ml_df[ml_df["pumping_mm"] > 0]
    dv_data = pd.get_dummies(ml_df, columns=['HSG'])
    y = dv_data['pumping_mm']
    X = dv_data.drop(columns=drop_attrs)
    print('Num predictors:', X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    scoring_metrics = {
        'r2': 'r2',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
        'neg_root_mean_squared_error_percent': make_scorer(
            neg_root_mean_squared_error_percent,
            greater_is_better=False
        ),
        'neg_mean_absolute_error_percent': make_scorer(
            neg_mean_absolute_error_percent,
            greater_is_better=False
        ),
        'coef_var': make_scorer(
            coef_var,
            greater_is_better=False
        )
    }
    for model_name in model_dict.keys():
        print('\nSearching best params for {}...'.format(model_name))
        model = model_dict[model_name]
        model_grid = RandomizedSearchCV(
            estimator=model, param_distributions=param_dict[model_name],
            scoring=scoring_metrics, n_jobs=-1, cv=5, refit=scoring_metrics['neg_root_mean_squared_error'],
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
        imp_df.to_csv(f'F_IMP_{site}.csv', index=False)
        y_pred_train = np.abs(model.predict(X_train))
        print('Training+Validation metrics...')
        r2, mae, rmse, cv = get_prediction_stats(y_train, y_pred_train)
        print(f'R2: {r2}, RMSE: {rmse}%, MAE: {mae}%, CV: {cv}%')
        y_pred_test = np.abs(model.predict(X_test))
        print('Test metrics...')
        r2, mae, rmse, cv = get_prediction_stats(y_test, y_pred_test)
        print(f'R2: {r2}, RMSE: {rmse}%, MAE: {mae}%, CV: {cv}%')
        # perm_scorer = scoring_metrics['neg_root_mean_squared_error_percent']
        # train_result = permutation_importance(
        #     model, X_train, y_train, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=perm_scorer
        # )
        # test_results = permutation_importance(
        #     model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=perm_scorer
        # )
        # sorted_importances_idx = train_result.importances_mean.argsort()
        # train_importances = pd.DataFrame(
        #     train_result.importances[sorted_importances_idx].T,
        #     columns=X.columns[sorted_importances_idx],
        # )
        # test_importances = pd.DataFrame(
        #     test_results.importances[sorted_importances_idx].T,
        #     columns=X.columns[sorted_importances_idx],
        # )
        # train_importances = train_importances[train_importances.columns[-5:]]
        # test_importances = test_importances[test_importances.columns[-5:]]
        # avg_train_rmse = train_importances[train_importances.columns[-1]].mean()
        # avg_test_rmse = test_importances[test_importances.columns[-1]].mean()
        # print(f'Avg train rmse increase: {avg_train_rmse}%')
        # print(f'Avg test rmse increase: {avg_test_rmse}%')
        # for name, importances in zip(["train", "test"], [train_importances, test_importances]):
        #     plt.figure(figsize=(10, 6))
        #     plt.rcParams.update({'font.size': 12})
        #     ax = importances.plot.box(vert=False, whis=10)
        #     ax.set_xlabel("Increase in RMSE (%)")
        #     ax.axvline(x=0, color="k", linestyle="--")
        #     ax.figure.tight_layout()
        #     plt.savefig(f'{model_name}_{name}_PI_{site}.png', dpi=600)
        #     plt.clf()
    # create_cv_files(dv_data, drop_attrs, model)


if __name__ == '__main__':
    ml_data_df = prepare_ml_data(site='dv')
    ml_data_df = process_dv_data(ml_data_df)
    build_ml_model(ml_data_df, site='dv')

    # ml_data_df = prepare_ml_data(site='hb')
    # ml_data_df = process_hb_data(ml_data_df)
    # HB, Oregon ML model for future use when there is sufficient pumping data available
    # build_ml_model(ml_data_df, site='hb')

