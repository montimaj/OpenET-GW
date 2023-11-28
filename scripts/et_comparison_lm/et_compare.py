import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_et(site='dv'):
    et_data_dict= {
        'ensemble': 'OpenET ensemble',
        'disalexi': 'ALEXI/disALEXI',
        'eemetric': 'eeMETRIC',
        'geesebal': 'geeSEBAL',
        'pt_jpl': 'PT-JPL',
        'sims': 'SIMS',
        'ssebop': 'SSEBop'
    }
    df = pd.read_csv(f'../machine_learning/{site}_joined_ml_pumping_data.csv')
    net_et_factor = 'pumping_net_et_ensemble_factor_annual'
    if site == 'hb':
        df = df[~df.fid.isin(['15', '533_1102', '1210_1211', '1329', '1539_1549_1550', '1692'])]
    et_df_all = df[df[net_et_factor] < 1.5]
    et_df_all = et_df_all[et_df_all[net_et_factor] > 0.5]
    et_df_all = et_df_all[et_df_all["pumping_mm"] > 0]
    et_df_all = et_df_all.rename(columns={'year': 'Year', 'pumping_mm': 'GP'})
    et_cols = [f'annual_et_{et_data}_mm' for et_data in et_data_dict.keys()]
    new_et_cols = [et_data_dict[et_data] for et_data in et_data_dict.keys()]
    new_et_dict = {}
    for idx, et_col in enumerate(et_cols):
        new_et_dict[et_col] = new_et_cols[idx]
    select_cols = et_cols + ['Year']
    et_df = et_df_all[select_cols]
    et_df = et_df.rename(columns=new_et_dict)
    et_df = et_df.groupby('Year').sum()
    et_df = et_df.reset_index()

    dfm = et_df.melt('Year', var_name='ET model', value_name='Total actual ET depth (mm)')
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size': 12})
    plt.ticklabel_format(style='sci', axis='y')
    sns.pointplot(
        data=dfm,
        x='Year',
        y='Total actual ET depth (mm)',
        hue='ET model'
    )
    plt.savefig(f'{site}_et_comp_plot.png', bbox_inches='tight', dpi=400)

    et_cols = [f'annual_net_et_{et_data}_mm' for et_data in et_data_dict.keys()]
    new_et_dict = {}
    for idx, et_col in enumerate(et_cols):
        new_et_dict[et_col] = new_et_cols[idx]
    select_cols = et_cols + ['Year', 'GP']
    et_df = et_df_all[select_cols]
    et_df = et_df.rename(columns=new_et_dict)
    et_df = et_df.groupby('Year').sum()
    et_df = et_df.reset_index()

    dfm = et_df.melt('Year', var_name='Net ET or GP', value_name='Total depth (mm)')
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size': 12})
    plt.ticklabel_format(style='sci', axis='y')
    sns.pointplot(
        data=dfm,
        x='Year',
        y='Total depth (mm)',
        hue='Net ET or GP'
    )
    plt.savefig(f'{site}_net_et_comp_plot.png', bbox_inches='tight', dpi=400)




if __name__ == '__main__':
    compare_et(site='dv')
    compare_et(site='hb')
