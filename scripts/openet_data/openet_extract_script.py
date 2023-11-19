
#############################################################################
#                           Diamond Valley
###############################################################################

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.geometry import Point
import json
import pandas as pd
import ee

ee.Initialize()

# General Input Variables
year_list = [2018, 2019, 2020, 2021, 2022]

# Effective precip factor
prz_factor_df = pd.read_csv(r'..\et-demands\harney_basin\effective_precip_fraction.csv')
prz_factor_table = prz_factor_df.set_index('Year')['P_rz_fraction'].to_dict()

# Import shapefile into featureCollection
gdf = gpd.read_file(r'..\gis\diamond_valley\dv_field_shp\153_DiamondValley_filtered.shp', crs="EPSG:26911")
gdf = gdf.to_crs("EPSG:4326")
geo_json = gdf.to_json()
fc = ee.FeatureCollection(json.loads(geo_json))

# Import Datasets
openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")

# Add area to fc
def add_area(fc):
    area = fc.area()
    return fc.set({'area_m2': area})
fc = fc.map(add_area)

df_list = []
for year in year_list:
    print(year)
    
    # Precip factor
    prz_factor = prz_factor_table[year]
    
    # Calendar year based
    # Build OpenET image
    an_openet_img = openet_ic.select('et_ensemble_mad')\
                          .filterDate(f'{year}-01-01', f'{year+1}-01-01')\
                          .sum()\
                          .rename('annual_et_mm')
    
    # Build gridMET image
    an_gridmet_img = gridmet_ic.select('pr')\
                            .filterDate(f'{year}-01-01', f'{year+1}-01-01')\
                            .sum()\
                            .rename('annual_precip_mm')
                            
    an_eff_gridmet_img = an_gridmet_img.multiply(prz_factor).rename('annual_precip_eff_mm')  
    
    data_img = an_openet_img.addBands(an_gridmet_img).addBands(an_eff_gridmet_img)
    
    # Filter Over Feature Collection
    data_fc = data_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=1)
    
    # Dowload table
    url = data_fc.getDownloadURL('csv', ['dri_field_', 'area_m2',
                                          'annual_et_mm', 'annual_precip_mm',
                                          'annual_precip_eff_mm'])
    df = pd.read_csv(url)
    df['YEAR'] = year
    df['eff_factor'] = prz_factor
    
    # Append to df
    df_list.append(df)

final_df = pd.concat(df_list)
final_df.to_csv('dv_openet_data_2018_2022.csv', index=False)   
    
    
# ###############################################################################
# #                           Harney Basin
# ###############################################################################

# import os
# os.environ['USE_PYGEOS'] = '0'
# import geopandas as gpd
# from shapely.geometry import Point
# import json
# import pandas as pd
# import ee

# ee.Initialize()

# # General Input Variables
# year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

# # Effective precip factor
# prz_factor_df = pd.read_csv(r'..\et-demands\harney_basin\effective_precip_fraction.csv')
# prz_factor_table = prz_factor_df.set_index('Year')['P_rz_fraction'].to_dict()

# # Import shapefile into featureCollection
# gdf = gpd.read_file(r'..\gis\harney_basin\harney_fields_2016_gw_well_wu_report_Merge.shp', crs="EPSG:26911")
# gdf = gdf.to_crs("EPSG:4326")
# geo_json = gdf.to_json()
# fc = ee.FeatureCollection(json.loads(geo_json))

# # Import Datasets
# openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
# gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")

# # Add area to fc
# def add_area(fc):
#     area = fc.area()
#     return fc.set({'area_m2': area})
# fc = fc.map(add_area)

# df_list = []
# for year in year_list:
#     print(year)
    
#     # Precip factor
#     prz_factor = prz_factor_table[year]
    
#     # Calendar year based
#     # Build OpenET image
#     an_openet_img = openet_ic.select('et_ensemble_mad')\
#                           .filterDate(f'{year}-01-01', f'{year+1}-01-01')\
#                           .sum()\
#                           .rename('annual_et_mm')
    
#     # Build gridMET image
#     an_gridmet_img = gridmet_ic.select('pr')\
#                             .filterDate(f'{year}-01-01', f'{year+1}-01-01')\
#                             .sum()\
#                             .rename('annual_precip_mm')
                            
#     an_eff_gridmet_img = an_gridmet_img.multiply(prz_factor).rename('annual_precip_eff_mm')  
    
#     data_img = an_openet_img.addBands(an_gridmet_img).addBands(an_eff_gridmet_img)
    
#     # Filter Over Feature Collection
#     data_fc = data_img.reduceRegions(
#         collection=fc,
#         reducer=ee.Reducer.mean(),
#         scale=30,
#         tileScale=1)
    
#     # Dowload table
#     url = data_fc.getDownloadURL('csv', ['Field_ID', 'area_m2',
#                                          'annual_et_mm', 'annual_precip_mm',
#                                          'annual_precip_eff_mm'])
#     df = pd.read_csv(url)
#     df['YEAR'] = year
#     df['eff_factor'] = prz_factor
    
#     # Append to df
#     df_list.append(df)

# final_df = pd.concat(df_list)
# final_df.to_csv('hb_openet_data_2016_2022.csv', index=False)   
        
        