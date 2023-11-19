# Purpose: To process all orginal shapefiles to prepare them for spatial joining.
# Author: Thomas Ott
# Date: Sometime in mid 2023

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd

#------------------------------------------------------------
#                    Process All POU Shapefile
#------------------------------------------------------------

# Import the orginal POU shapefile of all nevada POUs
all_pou_shp = gpd.read_file(r'dv_pou_shp\POU_Permits_All_dv.shp')

# Filter for only irrigation POU types
mou_list = ['IRR', 'IRD']
all_pou_shp = all_pou_shp.loc[all_pou_shp['mou'].isin(mou_list)].copy()

## Filter to Diamond Valley ##
# Get all single APP groups (this was produced from 2020 pumping data and manualy modified)
single_app_df = pd.read_excel(r'dv_pou_lists\app_groups_single.xlsx')

# Get multi groups (this was produced from 2020 pumping data and manualy modified)
multi_app_df = pd.read_excel(r'dv_pou_lists\app_groups_multiple.xlsx')
app_groups_df = pd.concat([multi_app_df,single_app_df], ignore_index=True)

# Cycle through each row and get list of lists
app_list = []
for row in app_groups_df.iterrows():
    # Try loop is needed to convert floats to integers but keep strings
    try:
        sub_list = list(row[1].dropna().values.astype(int).astype(str))
    except:
        sub_list = list(row[1].dropna().values.astype(str))
    
    # Remove leading and folowing spaces
    sub_list = [app.strip() for app in sub_list]
    
    # Append to main list
    app_list.append(sub_list)

# Loop through and merge each group into a single geometry
final_df_list = []
x=1
for app_group in app_list:
    sub_df = all_pou_shp[all_pou_shp.app.isin(app_group)]
    geometry = sub_df.unary_union
    
    # Convert lists into underscore separated objects
    app_group.sort()
    sub_app_list = list(sub_df.app.values)
    sub_app_list.sort()
    
    all_app_str = '_'.join(app_group)  
    sub_app_str = '_'.join(sub_app_list)
    
    final_df_list.append({'dri_id': f'dri_pou_group_id_{x}',
                          'all_app': all_app_str,
                          'sub_app': sub_app_str,
                          'geometry': geometry})
    x+=1

# Concatonate all dicts into geodataframe
final_df = gpd.GeoDataFrame(final_df_list)

# Build a negative buffer
final_df['geometry'] = final_df['geometry'].buffer(-50)

final_df.to_file(
    r"dv_pou_shp/POU_Permits_Diamond_Valley_Irrigation_Meter_merged_buffer.shp",
    driver='ESRI Shapefile',
    crs='EPSG:26911')


#------------------------------------------------------------
#               Process DRI field polygons 
#------------------------------------------------------------

# Import Field polygon shapefile 
all_fields_shp = gpd.read_file(r'dv_field_shp\153_DiamondValley.shp')
# Add an area field
all_fields_shp['area'] = all_fields_shp.geometry.area
# Filter for areas greater than 300_000 m2
all_fields_shp = all_fields_shp.loc[all_fields_shp['area'] > 300_000, :]
all_fields_shp = all_fields_shp.reset_index(drop=True)
all_fields_shp['FID'] = all_fields_shp.index.values
# Remove from specific list
removal_list = pd.read_excel(r'dv_pou_lists\dv_field_polygon_removal_list.xlsx')
removal_list = removal_list.FID.values
all_fields_shp = all_fields_shp.loc[~all_fields_shp['FID'].isin(removal_list), :]
all_fields_shp['dri_field_id'] = all_fields_shp['FID'] 
all_fields_shp = all_fields_shp.loc[:, ['dri_field_id', 'geometry']]

all_fields_shp.to_file(
    r"dv_field_shp/153_DiamondValley_filtered.shp",
    driver='ESRI Shapefile',
    crs='EPSG:26911')

#------------------------------------------------------------
#                           Spatial Join 
#------------------------------------------------------------
final_df_simp = gpd.read_file("dv_pou_shp/POU_Permits_Diamond_Valley_Irrigation_Meter_merged_buffer.shp")

# Join based on spatial extent
join_left_df = final_df_simp.sjoin(all_fields_shp, how="left")
join_left_df = join_left_df.loc[:, ['dri_id','all_app','sub_app','dri_field_id']]
join_left_df['dri_field_id'] = join_left_df['dri_field_id'].astype(str)

# Get main df
main_join_df = join_left_df.loc[:, ['dri_id', 'all_app', 'sub_app']].drop_duplicates()

# goup by dri_group_id
def join_func(my_list):
    return '_'.join(list(my_list))

dri_field_id = join_left_df.groupby(by="dri_id")['dri_field_id'].apply(join_func).reset_index(name='dri_field_id')
main_join_df = pd.merge(main_join_df, dri_field_id, on='dri_id')
main_join_df = main_join_df.astype(str)
# saving this to the joined data folder since this is the final csv
main_join_df.to_csv(r'..\..\joined_data\dv_field_pou_id_join.csv', index=False)







