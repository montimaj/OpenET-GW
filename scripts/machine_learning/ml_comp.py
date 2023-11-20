# Compare machine learning models in Diamond Valley, Nevada
# Author: Sayantan Majumdar (sayantan.majumdar@dri.edu)

import os

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import json
import pandas as pd
import ee

ee.Initialize()

# General Input Variables
year_list = [2018, 2019, 2020, 2021, 2022]

# Effective precip factor
prz_factor_df = pd.read_csv('../et-demands\harney_basin\effective_precip_fraction.csv')
prz_factor_table = prz_factor_df.set_index('Year')['P_rz_fraction'].to_dict()

# Import shapefile into featureCollection
gdf = gpd.read_file('../gis/diamond_valley/dv_field_shp/153_DiamondValley_filtered.shp', crs="EPSG:26911")
gdf = gdf.to_crs("EPSG:4326")
geo_json = gdf.to_json()
fc = ee.FeatureCollection(json.loads(geo_json))

# Import Datasets
openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")

def add_area(fc):
    area = fc.area()
    return fc.set({'area_m2': area})
fc = fc.map(add_area)

df_list = []
for year in year_list:
    print(year)

    # Build gridMET image
    gridmet_img = gridmet_ic.select('tmmx') \
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
    gridmet_bands = [
        gridmet_tmmn,
        gridmet_eto,
        gridmet_etr,
        gridmet_vpd,
        gridmet_vs,
        gridmet_rmax,
        gridmet_rmin
    ]
    gridmet_band_names = [
        'annual_tmmx_K',
        'annual_tmmn_K',
        'annual_eto_mm',
        'annual_etr_mm',
        'annual_vpd_kPa',
        'annual_vs_mps',
        'annual_rmax',
        'annual_rmin'
    ]
    for gb in gridmet_bands:
        gridmet_img = gridmet_img.addBands(gb)
    data_img = gridmet_img

    # Filter Over Feature Collection
    data_fc = data_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=1)

    # Dowload table
    url = data_fc.getDownloadURL('csv', gridmet_band_names)
    df = pd.read_csv(url)
    df['YEAR'] = year

    # Append to df
    df_list.append(df)

final_df = pd.concat([dv_df, pd.concat(df_list)])
final_df.to_csv('dv_ml_data_2018_2022.csv', index=False)