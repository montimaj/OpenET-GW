import pandas as pd

import ee
ee.Initialize()

region = ee.FeatureCollection("USGS/WBD/2017/HUC12").filter(ee.Filter.eq('huc12', '160600051503')).first()
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select('pr')

date_list = [f'{year_val:02d}-{month:02d}-01' for year_val in range(2000, 2021) for month in range(1, 13)]

dict_list = []
for date in date_list:
    print(date)
    
    # get year and month
    year = date.split('-')[0]
    month = date.split('-')[1]
    
    # Get date window
    start_date = ee.Date(date)
    end_date = start_date.advance(1, 'month')
       
    # Filter gridmet data
    lan_id_mask = ee.Image(f'users/LANID/LANID/Output/CONUS/lanid1997_2017/lanid{year}')
    
    # gridmet image
    gridmet_img = gridmet.filterDate(start_date, end_date).sum().updateMask(lan_id_mask)
    
    # reduce region
    fc_out = gridmet_img.reduceRegion(ee.Reducer.mean(), region.geometry(), 30)
    
    dict_out = fc_out.getInfo()
    dict_out['year'] = year
    dict_out['month'] = month
    
    dict_list.append(dict_out)
    
output_df = pd.DataFrame(dict_list)

output_df.to_csv('gridmet_precip.csv', index=False)
