import pandas as pd

# year_list = [2019, 2020, 2021, 2022]
join_table = pd.read_csv('dv_field_pou_id_join.csv')

# Read ET and Pumping tables
# ET table
et_df = pd.read_csv('..\openet_data\dv_openet_data_2018_2022.csv')
pumping_df = pd.read_csv(r'..\pumping_data\Diamond Valley Data for Meters Database Pumpage Report.csv')


# Process ET data (get volume in m3)
et_df['et_m3'] = et_df['wy_et_mm']*et_df['area_m2']/1000
et_df['precip_eff_m3'] = et_df['wy_precip_mm']*et_df['area_m2']/1000*0.75
et_df['dri_field_'] = et_df['dri_field_'].astype(str)
et_df['count_et'] = 1

# Process Pumping data
pumping_df = pumping_df.loc[:, ['App', 'UsageYear', 'UsageAmountAcreFT']]
pumping_df['pumping_m3'] = pumping_df['UsageAmountAcreFT']*1233.48 # acre-ft to m3
pumping_df['count_pump'] = 1

year = 2020
dri_field_id = '50'
sub_app = '21844'


# Extract both lists
dri_id_list = dri_field_id.split('_')
app_list = sub_app.split('_')

# Extract ET data
et_df_sub = et_df.loc[(et_df['dri_field_'].isin(dri_id_list)) & (et_df['YEAR'] == year)]
   
# Extract all app values
pumping_df_sub = pumping_df.loc[(pumping_df['App'].isin(app_list)) & (pumping_df['UsageYear'] == year)]


print(pumping_df_sub)
print(et_df_sub)

