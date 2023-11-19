
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

# Import shapefile into featureCollection
gdf = gpd.read_file(r'..\gis\diamond_valley\dv_field_shp\153_DiamondValley_filtered_all.shp', crs="EPSG:26911")
gdf = gdf.to_crs("EPSG:4326")
geo_json = gdf.to_json()
fc = ee.FeatureCollection(json.loads(geo_json))

# Import Datasets
openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
irr_ic = ee.ImageCollection('users/LANID/LANID/Output/CONUS/lanid1997_2017')


# Add area to fc
def add_area(fc):
    area = fc.area()
    return fc.set({'area_m2': area})
fc = fc.map(add_area)

df_list = []
for year in year_list:
    print(year)
    
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
    
    # Water Year based
    # Build OpenET image
    wy_openet_img = openet_ic.select('et_ensemble_mad')\
                          .filterDate(f'{year-1}-10-01', f'{year}-10-01')\
                          .sum()\
                          .rename('wy_et_mm')
    
    # Build gridMET image
    wy_gridmet_img = gridmet_ic.select('pr')\
                            .filterDate(f'{year-1}-10-01', f'{year}-10-01')\
                            .sum()\
                            .rename('wy_precip_mm')   
    
    # Irrigation Image
    if year in [2021, 2022]:
        year_irr = 2020
    else:
        year_irr = year
    
    irr_img = irr_ic.filter(ee.Filter.eq('system:index', f'lanid{year_irr}'))\
        .first()\
        .unmask(0)\
        .rename('irrigated')
    
    data_img = an_openet_img.addBands(an_gridmet_img).addBands(wy_openet_img)\
        .addBands(wy_gridmet_img).addBands(irr_img)
    
    # Filter Over Feature Collection
    data_fc = data_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=1)
    
    # Dowload table
    url = data_fc.getDownloadURL('csv', ['OPENET_ID', 'field_id', 'area_m2',
                                          'annual_et_mm', 'annual_precip_mm',
                                          'wy_et_mm', 'wy_precip_mm',
                                          'irrigated'])
    df = pd.read_csv(url)
    df['YEAR'] = year
    
    # Append to df
    df_list.append(df)

et_df = pd.concat(df_list)
et_df.to_csv('dv_openet_data_2018_2022_all_fields.csv', index=False)   

# ###############################################################################
# #                               NDVI
# ###############################################################################

# # from google.cloud import storage
# import requests
# import pandas as pd

# requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
# Access_Token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY4OTI2MTAyNSwianRpIjoiZDU5OTllYzMtOGM1My00ZGUwLTg2NTUtMjg1YmYzODFjYWU2IiwibmJmIjoxNjg5MjYxMDI1LCJ0eXBlIjoiYWNjZXNzIiwic3ViIjoiYTdHZmE2cnlMYmdWZEFQTENDZElIdUYxV2l3MiIsImV4cCI6MTcyMDc5NzAyNSwicm9sZXMiOiJ1c2VyIiwidXNlcl9pZCI6ImE3R2ZhNnJ5TGJnVmRBUExDQ2RJSHVGMVdpdzIifQ.0DRSUWPS3CPxxPVlqTqzYf9U0aNuCQ6s_Yaf_JkfUlE'

# df_list = []
# for year in range(2018, 2023):
#     params = {
#         'dataset': 'LANDSAT_SR',
#         'variable': 'NDVI',
#         'temporal_statistic': 'max',
#         'start_date': f'{year}-05-01',
#         'end_date': f'{year}-10-31',  
#         'area_reducer': 'mean',
#         'asset_id': 'users/jmjthomasott/work_projects/nevada-gw-pumping/153_DiamondValley_filtered_all',
#         'filter_by': 'field_id',
#         'sub_choices': str(list(range(0, 491)))
#     }
    
#     url = 'https://api.climateengine.org/zonal_stats/values/custom_asset'
#     r = requests.get(url, params=params, headers={'Authorization': Access_Token}, verify=False)
#     response = r.json()
#     print(year)
    
#     # Convert Json file to csv
#     flat_list = [item for sublist in response for item in sublist]
#     df = pd.DataFrame(flat_list)
#     df = df.replace('Name not found!', 0)
#     df['YEAR'] = year
#     df_list.append(df.loc[:,['field_id', 'YEAR', 'NDVI']])

# ndvi_df = pd.concat(df_list)
# # ndvi_df.to_csv('dv_ndvi_data_2018_2022_all_fields.csv', index=False)

# # # Merge dataframes
# final_df = pd.merge(et_df, ndvi_df, how='inner', on=['field_id', 'YEAR'])
# final_df.to_csv('dv_data_2018_2022_all_fields.csv')







