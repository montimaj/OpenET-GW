import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_et(site='dv', unit='mm'):
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
    select_cols = et_cols + ['Year'] + ['area_m2']
    et_df = et_df_all[select_cols]
    et_df = et_df.rename(columns=new_et_dict)
    year_list = et_df_all.Year.unique()
    area_weight_et_df = pd.DataFrame()
    for et_col in new_et_dict.values():
        for year in year_list:
            new_df = pd.DataFrame()
            sub_df = et_df[et_df.Year == year].copy(deep=True)
            tot_area = sub_df['area_m2'].sum()
            sub_df[f'{et_col}_vol'] = sub_df[et_col] * sub_df['area_m2'] / tot_area
            new_df['Area-weighted mean actual ET depth (mm)'] = sub_df[[f'{et_col}_vol']].sum()
            new_df['Mean actual ET depth (mm)'] = sub_df[et_col].mean()
            new_df['ET model'] = et_col
            new_df['Year'] = year
            area_weight_et_df = pd.concat([area_weight_et_df, new_df])
    factor = 1
    if unit == 'ft':
        factor = 304.8
    area_weight_et_df[f'Area-weighted mean actual ET depth ({unit})'] = area_weight_et_df[
                                                                 'Area-weighted mean actual ET depth (mm)'] / factor
    area_weight_et_df[f'Mean actual ET depth ({unit})'] = area_weight_et_df['Mean actual ET depth (mm)'] / factor
    area_weight_et_df.to_csv(f'Area_Weighted_ET_{unit}.csv', index=False)


    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size': 12})
    plt.ticklabel_format(style='sci', axis='y')
    sns.pointplot(
        data=area_weight_et_df,
        x='Year',
        y=f'Area-weighted mean actual ET depth ({unit})',
        hue='ET model'
    )
    plt.savefig(f'{site}_et_comp_plot_{unit}.png', bbox_inches='tight', dpi=400)

    et_cols = [f'annual_net_et_{et_data}_mm' for et_data in et_data_dict.keys()]
    new_et_dict = {}
    for idx, et_col in enumerate(et_cols):
        new_et_dict[et_col] = new_et_cols[idx]
    select_cols = et_cols + ['Year', 'GP', 'area_m2']
    et_df = et_df_all[select_cols]
    et_df = et_df.rename(columns=new_et_dict)

    area_weight_net_et_df = pd.DataFrame()
    et_cols = list(new_et_dict.values()) + ['GP']
    for et_col in et_cols:
        for year in year_list:
            new_df = pd.DataFrame()
            sub_df = et_df[et_df.Year == year].copy(deep=True)
            tot_area = sub_df['area_m2'].sum()
            sub_df[f'{et_col}_vol'] = sub_df[et_col] * sub_df['area_m2'] / tot_area
            new_df['Area-weighted mean depth (mm)'] = sub_df[[f'{et_col}_vol']].sum()
            new_df['Mean depth (mm)'] = sub_df[et_col].mean()
            new_df['Net ET or GP'] = et_col
            new_df['Year'] = year
            area_weight_net_et_df = pd.concat([area_weight_net_et_df, new_df])
    area_weight_net_et_df[f'Area-weighted mean depth ({unit})'] = area_weight_net_et_df[
                                                                 'Area-weighted mean depth (mm)'] / factor
    area_weight_net_et_df['Mean depth (ft)'] = area_weight_net_et_df['Mean depth (mm)'] / factor
    area_weight_net_et_df.to_csv('Area_Weighted_Net_ET_{unit}.csv', index=False)

    plt.figure(figsize=(25, 12))
    plt.rcParams.update({'font.size': 24})
    plt.ticklabel_format(style='sci', axis='y')
    ax = sns.pointplot(
        data=area_weight_net_et_df,
        x='Year',
        y=f'Area-weighted mean depth ({unit})',
        hue='Net ET or GP'
    )
    ax.legend(loc='upper left', ncol=2, title='Net ET or GP')
    plt.savefig(f'{site}_net_et_comp_plot_{unit}.png', bbox_inches='tight', dpi=400)
    plt.clf()

    # area_weight_net_et_df = area_weight_net_et_df[area_weight_net_et_df['Net ET or GP'] != 'GP']
    # diff_mean = area_weight_net_et_df['Area-weighted mean depth (mm)'] - area_weight_net_et_df['Mean depth (mm)']
    # plt.rcParams.update({'font.size': 12})
    # sns.displot(x=diff_mean)
    # plt.xlabel('Area-weighted mean depth - Mean depth (mm)')
    # plt.savefig(f'{site}_net_et_diff_plot_mm.png', bbox_inches='tight', dpi=400)
    # plt.clf()
    #
    # diff_mean = area_weight_net_et_df['Area-weighted mean depth (ft)'] - area_weight_net_et_df['Mean depth (ft)']
    # plt.rcParams.update({'font.size': 12})
    # sns.displot(x=diff_mean)
    # plt.xlabel('Area-weighted mean depth - Mean depth (ft)')
    # plt.savefig(f'{site}_net_et_diff_plot_ft.png', bbox_inches='tight', dpi=400)




if __name__ == '__main__':
    compare_et(site='dv')
    compare_et(site='hb')
