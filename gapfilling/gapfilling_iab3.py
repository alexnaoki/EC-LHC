import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class gapfilling_iab3:
    def __init__(self, ep_path, lf_path, iab2_path, iab1_path):
        self.iab3EP_path = pathlib.Path(ep_path)
        self.iab3LF_path = pathlib.Path(lf_path)

        self.iab2_path = pathlib.Path(iab2_path)

        self.iab1_path = pathlib.Path(iab1_path)

        self._read_files()

    def _read_files(self):
        self.iab3EP_files = self.iab3EP_path.rglob('eddypro*p08*full*.csv')
        self.iab3LF_files = self.iab3LF_path.rglob('TOA5*.flux*.dat')
        self.iab2_files = self.iab2_path.rglob('*.dat')
        self.iab1_files = self.iab1_path.rglob('*Table1*.dat')

        ep_columns =  ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
               'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed',
               'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg']
        lf_columns = ['TIMESTAMP', 'CO2_sig_strgth_mean','H2O_sig_strgth_mean','Rn_Avg','Rs_incoming_Avg', 'Rs_outgoing_Avg',
                      'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg','Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)',
                      'precip_Tot']
        iab3EP_dfs = []
        print('Reading IAB3_EP files...')
        for file in self.iab3EP_files:
            iab3EP_dfs.append(pd.read_csv(file,skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date','time']}, keep_date_col=True, usecols=ep_columns))
        self.iab3EP_df = pd.concat(iab3EP_dfs)
        print(f'# IAB3_EP: {len(iab3EP_dfs)}')

        iab3LF_dfs = []
        print('Reading IAB3_LF files...')
        for file in self.iab3LF_files:
            iab3LF_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP'], usecols=lf_columns))
        self.iab3LF_df = pd.concat(iab3LF_dfs)
        print(f'# IAB3_LF: {len(iab3LF_dfs)}')

        iab2_dfs = []
        print('Reading IAB2 files...')
        for file in self.iab2_files:
            iab2_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab2_df = pd.concat(iab2_dfs)
        print(f'# IAB2: {len(iab2_dfs)}')

        iab1_dfs = []
        print('Reading IAB1 files...')
        for file in self.iab1_files:
            iab1_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab1_df = pd.concat(iab1_dfs)
        print(f'# IAB1: {len(iab1_dfs)}')

        iab_dfs = [self.iab3EP_df, self.iab3LF_df, self.iab2_df, self.iab1_df]

        for df in iab_dfs:
            print('Duplicatas: ',df.duplicated().sum())
            df.drop_duplicates(subset='TIMESTAMP', keep='first', inplace=True)
            df.reset_index(inplace=True)
            print('Verificacao de Duplicatas: ', df.duplicated().sum())

        self.iab3_df = pd.merge(left=self.iab3EP_df, right=self.iab3LF_df, on='TIMESTAMP', how='inner')

    def _applying_filters(self):
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)==2, 'flag_qaqc'] = 1
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)!=2, 'flag_qaqc'] = 0

        self.iab3_df.loc[self.iab3_df['precip_Tot']>0, 'flag_rain'] = 0
        self.iab3_df.loc[self.iab3_df['precip_Tot']==0, 'flag_rain'] = 1

        min_signalStr = 0.8
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']>=min_signalStr, 'flag_signalStr'] = 1
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']<min_signalStr, 'flag_signalStr'] = 0

    def dropping_bad_data(self):
        self._applying_filters()
        iab3_df_copy = self.iab3_df.copy()
        iab3_df_copy.loc[
            (iab3_df_copy['flag_qaqc']==0)|
            (iab3_df_copy['flag_rain']==0)|
            (iab3_df_copy['flag_signalStr']==0), 'ET'] = np.nan

        return iab3_df_copy


    def _adjacent_days(self, df,n_days=5):
        delta_days = [i for i in range(-n_days, n_days+1, 1)]
        df[f'timestamp_adj_{n_days}'] = df['TIMESTAMP'].apply(lambda x: [x + dt.timedelta(days=i) for i in delta_days])

    def mdv_test(self, n_days=5):
        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=['ET'], inplace=True)
        a, b = train_test_split(iab3_df_copy[['TIMESTAMP','ET']])

        date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                   end=iab3_df_copy['TIMESTAMP'].max(),
                                   freq='30min')
        df_date_range = pd.DataFrame({'TIMESTAMP':date_range})
        iab3_alldates = pd.merge(left=df_date_range, right=a, on='TIMESTAMP', how='outer')
        # iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')
        b.rename(columns={"ET":'ET_val_mdv'}, inplace=True)
        # print(b)
        iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')
        self._adjacent_days(df=iab3_alldates, n_days=n_days)

        # print(iab3_alldates)

        for i, row in iab3_alldates.loc[iab3_alldates['ET'].isna()].iterrows():
            iab3_alldates.loc[i, f'ET_mdv_{n_days}'] = iab3_alldates.loc[(iab3_alldates['TIMESTAMP'].isin(row[f'timestamp_adj_{n_days}']))&
                                                                         (iab3_alldates['ET'].notna()), 'ET'].mean()

        # print(iab3_alldates)

        iab3_metrics = iab3_alldates[['ET_val_mdv',f'ET_mdv_{n_days}']].copy()
        iab3_metrics.dropna(inplace=True)
        # print(iab3_metrics)
        print(mean_absolute_error(iab3_metrics['ET_val_mdv'], iab3_metrics[f'ET_mdv_{n_days}']))
        print(iab3_metrics.corr())

    def rfr_test(self):
        column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
        column_x_ET = column_x + ['ET']

        iab3 = self.iab3_df_copy.dropna(subset=column_x_ET)
        X = iab3[column_x]
        y = iab3['ET']

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        et_model_RFR = RandomForestRegressor(random_state=1, criterion='mae')
        et_model_RFR.fit(train_X, train_y)

        val_prediction_RFR = et_model_RFR.predict(val_X)
        mae_RFR = mean_absolute_error(val_y, val_prediction_RFR)


if __name__ == '__main__':
    gapfilling_iab3(ep_path=r'G:\Meu Drive\USP-SHS\Resultados_processados\EddyPro_Fase010203',
                    lf_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB3',
                    iab1_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB1\IAB1',
                    iab2_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB2\IAB2')
