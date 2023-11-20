import pandas as pd
import numpy as np

year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

# Read ET and Pumping tables
# ET table
et_df = pd.read_csv('../openet_data/hb_openet_data_2016_2022.csv')
pumping_df = pd.read_excel('../pumping_data/harney_basin/Harney_PumpingWell_FieldID_WUR_Relationship_2016_2022.xlsx', header=1)
prz_factor = pd.read_csv('../et-demands/harney_basin/effective_precip_fraction.csv')

# Build join table
join_table = pumping_df.loc[:,['WUR_Report_ID', 'FID_1', 'FID_2', 'FID_3']]
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
  

# Process ET data (get volume in m3)
# wy_et_mm, wy_precip_mm, annual_et_mm, annual_precip_mm
# et_df['wy_et_m3'] = et_df['wy_et_mm']*et_df['area_m2']/1000
# et_df['wy_precip_eff_m3'] = et_df['wy_precip_mm']*et_df['area_m2']*0.75/1000

et_df['annual_et_m3'] = et_df['annual_et_mm']*et_df['area_m2']/1000
et_df['annual_precip_eff_m3'] = et_df['annual_precip_eff_mm']*et_df['area_m2']/1000

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
        et_df_sub = et_df_sub.loc[:, ['area_m2', 'annual_et_m3', 'annual_precip_eff_m3']]
        et_dict = dict(et_df_sub.sum())
        
        # Extract all app values
        pumping_df_sub = pumping_df.loc[pumping_df['WUR_Report_ID']==app_id, [f'TWU_{year}']]*1233.48 # convert acre-ft to m3
        pumping_df_sub = pumping_df_sub.rename(columns={f'TWU_{year}':'pumping_m3'})
        pumping_dict = dict(pumping_df_sub.sum())
        
        # Pumping outlier
        outlier_df_sub = pumping_df.loc[pumping_df['WUR_Report_ID']==app_id, [f'Outlier_{year}']] # convert acre-ft to m3
        outlier_df_sub = outlier_df_sub.rename(columns={f'Outlier_{year}':'outlier'})
        outlier_dict = dict(outlier_df_sub.sum())
        
        # Check if meter was changed this year (remove if it was)
        method_df = pumping_df.loc[pumping_df['WUR_Report_ID']==app_id, [f'Meth_{year}']]
        method_df = method_df.rename(columns={f'Meth_{year}':'method'})
        method_dict = dict(method_df.sum())
            
        # merge dictionaries
        et_dict.update(pumping_dict)
        et_dict.update(method_dict)
        et_dict.update(outlier_dict)
        et_dict.update(id_)
        dict_list.append(et_dict)
        

final_df = pd.DataFrame(dict_list)

# Build addictional units
final_df['annual_net_et_mm'] = (final_df['annual_et_m3'] - final_df['annual_precip_eff_m3'])/final_df['area_m2'] * 1000
final_df['pumping_mm'] = final_df['pumping_m3']/final_df['area_m2'] * 1000
final_df['annual_precip_eff_mm'] = (final_df['annual_precip_eff_m3'])/final_df['area_m2'] * 1000

### Add factors ###
final_df['pumping_net_et_factor_annual'] = np.round(final_df['pumping_mm'] / final_df['annual_net_et_mm'], 1)


final_df = final_df.loc[:, ['WUR_Report_ID', 'fid', 
                            'year', 'method', 'area_m2',
                            'annual_precip_eff_mm',
                            'annual_net_et_mm', 'pumping_mm',
                            'pumping_net_et_factor_annual'
                            ]]

final_df.to_csv('hb_joined_et_pumping_data_all.csv', index=False)


