import pandas as pd
import numpy as np

year_list = [2018, 2019, 2020, 2021, 2022]
join_table = pd.read_csv('dv_field_pou_id_join.csv')

# Read ET and Pumping tables
# ET table
et_df = pd.read_csv('../openet_data/dv_openet_data_2018_2022.csv')
pumping_df = pd.read_csv('../pumping_data/diamond_valley/Diamond Valley Data for Meters Database Pumpage Report.csv')
prz_factor = pd.read_csv('../et-demands/diamond_valley/effective_precip_fraction.csv')

# Process ET data (get volume in m3)
# wy_et_mm, wy_precip_mm, annual_et_mm, annual_precip_mm
# et_df['wy_et_m3'] = et_df['wy_et_mm']*et_df['area_m2']/1000
# et_df['wy_precip_eff_m3'] = et_df['wy_precip_mm']*et_df['area_m2']*0.75/1000

et_df['annual_et_m3'] = et_df['annual_et_mm']*et_df['area_m2']/1000
et_df['annual_precip_eff_m3'] = et_df['annual_precip_eff_mm']*et_df['area_m2']/1000

et_df['dri_field_'] = et_df['dri_field_'].astype(str)
et_df['count_et'] = 1
 
# Process Pumping data
pumping_df = pumping_df.loc[pumping_df['Method'] == 'METER', :] # Filter to only metered data
pumping_df = pumping_df.loc[:, ['App', 'UsageYear', 'UsageAmountAcreFT', 'MeterDeactivated']]
pumping_df['pumping_m3'] = pumping_df['UsageAmountAcreFT']*1233.48 # acre-ft to m3
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
        # et_df_sub = et_df_sub.loc[:, ['area_m2', 'wy_et_m3', 'wy_precip_eff_m3', 'annual_et_m3', 'annual_precip_eff_m3', 'count_et']]
        et_df_sub = et_df_sub.loc[:, ['area_m2', 'annual_et_m3', 'annual_precip_eff_m3', 'count_et']]
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
        pumping_df_sub = pumping_df_sub.drop_duplicates() # grouped apps have the same pumping data
        pumping_dict = dict(pumping_df_sub.sum())
        
        # merge dictionaries
        et_dict.update(pumping_dict)
        et_dict.update(id_)
        et_dict.update(meter_dict)
        
        dict_list.append(et_dict)


final_df = pd.DataFrame(dict_list)

# Build addictional units
final_df['annual_net_et_mm'] = (final_df['annual_et_m3'] - final_df['annual_precip_eff_m3'])/final_df['area_m2'] * 1000
final_df['pumping_mm'] = final_df['pumping_m3']/final_df['area_m2'] * 1000
final_df['annual_precip_eff_mm'] = (final_df['annual_precip_eff_m3'])/final_df['area_m2'] * 1000

### Add factors ###
final_df['pumping_net_et_factor_annual'] = np.round(final_df['pumping_mm'] / final_df['annual_net_et_mm'], 1)


final_df = final_df.loc[:, ['dri_field_id', 'all_app', 
                            'year', 'area_m2',
                            'annual_precip_eff_mm',
                            'annual_net_et_mm', 'pumping_mm',
                            'pumping_net_et_factor_annual'
                            ]]

final_df.to_csv('dv_joined_et_pumping_data_all.csv', index=False)
    