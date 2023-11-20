import pandas as pd
from statistics import mean 
import os



# Get list of all files in data folder
file_list = os.listdir('annual_stats')


niwr = []
prz_df_list = []
for file in file_list:
    cell = file.split('_')[0]
    
    df = pd.read_csv(f'annual_stats/{file}', header=1)
    
    # Add cell to df
    df['cell'] = cell
    

    # Get net irrigation water requirement
    niwr.append(df.NIWR.mean())
    
    # Get all P_rz
    df_sub = df.loc[df.Year >= 2016,  ['Year', 'P_rz_fraction', 'P_rz', 'PPT', 'NIWR', 'ETpot', 'ETact', 'P_eft_fraction', 'P_eft']]
    
    prz_df_list.append(df_sub)
    

# NIWR
niwr_value = mean(niwr)

# Get average annual fractoin
prz_df = pd.concat(prz_df_list)

prz = prz_df.groupby('Year').mean()

prz.to_csv('effective_precip_fraction.csv')
