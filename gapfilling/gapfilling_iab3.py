import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf
import calendar
import math
import matplotlib.dates as mdates
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

from scipy.stats import f_oneway, shapiro, kstest, kruskal, friedmanchisquare, normaltest
from scipy.stats.mstats import kruskalwallis
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class gapfilling_iab3:
    def __init__(self, ep_path, lf_path, iab2_path, iab1_path, footprint_file):
        # File's Path
        self.iab3EP_path = pathlib.Path(ep_path)
        self.iab3LF_path = pathlib.Path(lf_path)
        self.iab2_path = pathlib.Path(iab2_path)
        self.iab1_path = pathlib.Path(iab1_path)

        self.footprint_file = pathlib.Path(footprint_file)

        self._read_files()

        # self._gagc()

    def _read_files(self):
        # Reading csv files
        self.iab3EP_files = self.iab3EP_path.rglob('eddypro*p08*full*.csv')
        self.iab3LF_files = self.iab3LF_path.rglob('TOA5*.flux*.dat')
        self.iab2_files = self.iab2_path.rglob('*.dat')
        self.iab1_files = self.iab1_path.rglob('*Table1*.dat')

        # self.footprint_file = self.footprint_path.rglob('classification_pixel_2018-10-05-00-30to2020-11-04-00-00_pf_80*')

        footprint_columns = ['TIMESTAMP','code03', 'code04', 'code09','code12','code15','code19','code20','code24','code25','code33']
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
        print(f"# IAB3_EP: {len(iab3EP_dfs)}\tInicio: {self.iab3EP_df['TIMESTAMP'].min()}\tFim: {self.iab3EP_df['TIMESTAMP'].max()}")

        iab3LF_dfs = []
        print('Reading IAB3_LF files...')
        for file in self.iab3LF_files:
            iab3LF_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP'], usecols=lf_columns))
        self.iab3LF_df = pd.concat(iab3LF_dfs)
        print(f"# IAB3_LF: {len(iab3LF_dfs)}\tInicio:{self.iab3LF_df['TIMESTAMP'].min()}\tFim: {self.iab3LF_df['TIMESTAMP'].max()}")

        iab2_dfs = []
        print('Reading IAB2 files...')
        for file in self.iab2_files:
            iab2_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab2_df = pd.concat(iab2_dfs)
        print(f"# IAB2: {len(iab2_dfs)}\tInicio: {self.iab2_df['TIMESTAMP'].min()}\tFim: {self.iab2_df['TIMESTAMP'].max()}")

        iab1_dfs = []
        print('Reading IAB1 files...')
        for file in self.iab1_files:
            iab1_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab1_df = pd.concat(iab1_dfs)
        print(f"# IAB1: {len(iab1_dfs)}\tInicio: {self.iab1_df['TIMESTAMP'].min()}\tFim: {self.iab1_df['TIMESTAMP'].max()}")

        iab_dfs = [self.iab3EP_df, self.iab3LF_df, self.iab2_df, self.iab1_df]

        print('Reading Footprint file...')
        self.footprint_df = pd.read_csv(self.footprint_file, parse_dates=['TIMESTAMP'], na_values=-9999, usecols=footprint_columns)
        self.footprint_df.drop_duplicates(subset='TIMESTAMP', inplace=True)
        print(f"Inicio: {self.footprint_df['TIMESTAMP'].min()}\tFim: {self.footprint_df['TIMESTAMP'].max()}")

        # Removing duplicated files based on 'TIMESTAMP'
        for df in iab_dfs:
            print('Duplicatas: ',df.duplicated().sum())
            df.drop_duplicates(subset='TIMESTAMP', keep='first', inplace=True)
            df.reset_index(inplace=True)
            print('Verificacao de Duplicatas: ', df.duplicated().sum())


        # Merging files from EddyPro data and LowFreq data
        self.iab3_df = pd.merge(left=self.iab3EP_df, right=self.iab3LF_df, on='TIMESTAMP', how='inner')

        # Merging EP and LF data with footprint data
        self.iab3_df = pd.merge(left=self.iab3_df, right=self.footprint_df, on='TIMESTAMP', how='inner')

        # print(self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==2020)&(self.iab3_df['TIMESTAMP'].dt.month==11),'ET'].describe())
        # print(self.iab3EP_df.loc[(self.iab3EP_df['TIMESTAMP'].dt.year==2020)&(self.iab3EP_df['TIMESTAMP'].dt.month==11),'ET'].describe())
        # print(self.iab3LF_df.loc[(self.iab3LF_df['TIMESTAMP'].dt.year==2020)&(self.iab3LF_df['TIMESTAMP'].dt.month==11)].describe())


        # Resampling IAB2
        self.iab2_df_resample = self.iab2_df.set_index('TIMESTAMP').resample('30min').mean()
        self.iab2_df_resample.reset_index(inplace=True)
        # print(self.iab2_df_resample)

    def _applying_filters(self):
        # Flag using Mauder and Foken (2004)
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)==2, 'flag_qaqc'] = 1
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)!=2, 'flag_qaqc'] = 0

        # Flag rain
        self.iab3_df.loc[self.iab3_df['precip_Tot']>0, 'flag_rain'] = 0
        self.iab3_df.loc[self.iab3_df['precip_Tot']==0, 'flag_rain'] = 1

        # Flag signal strength
        min_signalStr = 0.8
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']>=min_signalStr, 'flag_signalStr'] = 1
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']<min_signalStr, 'flag_signalStr'] = 0

        # Flag Footprint
        self.iab3_df['footprint_acceptance'] = self.iab3_df[['code03', 'code04']].sum(axis=1)/self.iab3_df[['code03','code04','code09','code12','code15','code19','code20','code24','code25','code33']].sum(axis=1)
        min_footprint = 0.8
        self.iab3_df.loc[self.iab3_df['footprint_acceptance']>=min_footprint, 'flag_footprint'] = 1
        self.iab3_df.loc[self.iab3_df['footprint_acceptance']<min_footprint, 'flag_footprint'] = 0

        # print(self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==2020)&(self.iab3_df['TIMESTAMP'].dt.month==11),'ET'].describe())

    def dropping_bad_data(self):
        # Apply filters
        self._applying_filters()
        # print('fdas')

        # Creating a copy and changing to 'nan' filtered values
        iab3_df_copy = self.iab3_df.copy()
        # print(iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.year==2020)&(iab3_df_copy['TIMESTAMP'].dt.month==11), 'ET'].describe())

        iab3_df_copy.loc[
            (iab3_df_copy['flag_qaqc']==0)|
            (iab3_df_copy['flag_rain']==0)|
            (iab3_df_copy['flag_signalStr']==0)|
            (iab3_df_copy['LE']<0), 'ET'] = np.nan

        iab3_df_copy.loc[
            (iab3_df_copy['flag_qaqc']==0)|
            (iab3_df_copy['flag_rain']==0)|
            (iab3_df_copy['flag_signalStr']==0)|
            (iab3_df_copy['LE']<0), 'LE'] = np.nan



        # print(iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.year==2020)&(iab3_df_copy['TIMESTAMP'].dt.month==11), 'ET'].describe())

        use_footprint = True
        if use_footprint:
            iab3_df_copy.loc[
                (iab3_df_copy['flag_footprint']==0), 'ET'] = np.nan
            iab3_df_copy.loc[
                (iab3_df_copy['flag_footprint']==0), 'LE'] = np.nan

        # print(iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.year==2020)&(iab3_df_copy['TIMESTAMP'].dt.month==11), 'ET'].describe())

        return iab3_df_copy

    def _adjacent_days(self, df,n_days=5):
        # Selecting datetime adjectent
        delta_days = [i for i in range(-n_days, n_days+1, 1)]
        df[f'timestamp_adj_{n_days}'] = df['TIMESTAMP'].apply(lambda x: [x + dt.timedelta(days=i) for i in delta_days])

    def _gagc(self):
        self.iab3_df.loc[(self.iab3_df['flag_qaqc']==0)|
                               (self.iab3_df['flag_rain']==0)|
                               (self.iab3_df['flag_signalStr']==0)|
                               (self.iab3_df['flag_footprint']==0), 'LE'] = np.nan

        self.iab3_df['psychrometric_kPa'] = 0.665*10**(-3)*self.iab3_df['air_pressure']/1000
        self.iab3_df['delta'] = 4098*(0.6108*np.e**(17.27*(self.iab3_df['air_temperature']-273.15)/((self.iab3_df['air_temperature']-273.15)+237.3)))/((self.iab3_df['air_temperature']-273.15)+237.3)**2
        self.iab3_df['VPD_kPa'] = (self.iab3_df['es']-self.iab3_df['e'])/1000
        self.iab3_df['LE_MJmh'] = self.iab3_df['LE']*3600/1000000
        self.iab3_df['Rn_Avg_MJmh'] = self.iab3_df['Rn_Avg']*3600/1000000
        self.iab3_df['shf_Avg_MJmh'] = self.iab3_df[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1)*3600/1000000

        self.iab3_df['ga'] = (self.iab3_df['wind_speed']/self.iab3_df['u*']**2)**(-1)
        self.iab3_df['gc'] = (self.iab3_df['LE_MJmh']*self.iab3_df['psychrometric_kPa']*self.iab3_df['ga'])/(self.iab3_df['delta']*(self.iab3_df['Rn_Avg_MJmh']-self.iab3_df['shf_Avg_MJmh'])+self.iab3_df['air_density']*3600*1.013*10**(-3)*self.iab3_df['VPD_kPa']*self.iab3_df['ga']-self.iab3_df['LE_MJmh']*self.iab3_df['delta']-self.iab3_df['LE_MJmh']*self.iab3_df['psychrometric_kPa'])

        # print(self.iab3_df['gc'].describe())
        self.iab3_df_gagc = self.iab3_df.set_index('TIMESTAMP').resample('1m').mean()[['ga','gc']]
        self.iab3_df_gagc.reset_index(inplace=True)
        # print(self.iab3_df_gagc)

    def _gagc_2(self):
        self._gagc()
        pm_inputs_iab3 = ['delta', 'Rn_Avg_MJmh', 'shf_Avg_MJmh', 'air_density', 'VPD_kPa', 'ga','LE_MJmh','psychrometric_kPa', 'gc', 'TIMESTAMP']
        pm_inputs_iab3_ET = pm_inputs_iab3 + ['ET']
        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=pm_inputs_iab3_ET, inplace=True)

        train, val = train_test_split(iab3_df_copy, shuffle=True)

        self.iab3_df_gagc_2 = train.set_index('TIMESTAMP').resample('1m').mean()[['ga','gc']]
        self.iab3_df_gagc_2.reset_index(inplace=True)

        # print(train[['ga','gc']].describe())
        # print(val[['ga','gc']].describe())
        # print(train)
        # print(val)

    def _adjusting_input_pm(self):
        # self.iab2_df_resample['delta'] = 4098*(0.6108*np.e**(17.27*self.iab2_df_resample['AirTC_Avg']/(self.iab2_df_resample['AirTC_Avg']+237.3)))/(self.iab2_df_resample['AirTC_Avg']+237.3)**2

        # self.iab2_df_resample['es'] = 0.6108*np.e**(17.27*self.iab2_df_resample['AirTC_Avg']/(self.iab2_df_resample['AirTC_Avg']+237.3))

        self.iab12_df = pd.merge(left=self.iab1_df[['TIMESTAMP','RH']],
                                 right=self.iab2_df_resample[['TIMESTAMP', 'AirTC_Avg','CNR_Wm2_Avg','G_Wm2_Avg']],
                                 on='TIMESTAMP', how='inner')

        self.iab12_df['delta'] = 4098*(0.6108*np.e**(17.27*self.iab12_df['AirTC_Avg']/(self.iab12_df['AirTC_Avg']+237.3)))/(self.iab12_df['AirTC_Avg']+237.3)**2
        self.iab12_df['es'] = 0.6108*np.e**(17.27*self.iab12_df['AirTC_Avg']/(self.iab12_df['AirTC_Avg']+237.3))
        self.iab12_df['ea'] = self.iab12_df['RH']/100*self.iab12_df['es']
        self.iab12_df['VPD'] = self.iab12_df['es']-self.iab12_df['ea']

        altitude = 790
        self.iab12_df['P'] = 101.3*((293-0.0065*altitude)/293)**5.26
        self.iab12_df['psychrometric_cte'] = 0.665*10**(-3)*self.iab12_df['P']

        self.iab12_df['air_density'] = 1.088

        self.iab12_df['Rn_Avg_MJmh'] = self.iab12_df['CNR_Wm2_Avg']*3600/1000000
        self.iab12_df['G_Avg_MJmh'] = self.iab12_df['G_Wm2_Avg']*3600/1000000
        ga = 0.051

        # gc_iab3 = []
        # for i in range(1, 13, 1):

        # no gc tem um problema que Rn é muito alto, deveria contabilizar somente ET (Latent portion)
        # self.iab12_df['ET_iab12'] = (self.iab12_df['delta']*self.iab12_df['Rn_Avg_MJmh']+3600*self.iab12_df['air_density']*1.013*10**(-3)*self.iab12_df['VPD']*ga)/()
        # for i in range(1, 13, 1):
        self.iab12_df['gc'] = (self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['psychrometric_cte']*ga)/(self.iab12_df['delta']*(self.iab12_df['Rn_Avg_MJmh']-self.iab12_df['G_Avg_MJmh'])+self.iab12_df['air_density']*3600*1.013*10**(-3)*self.iab12_df['VPD']*ga-self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['delta']-self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['psychrometric_cte'])

        self.iab12_df.loc[(self.iab12_df['gc']>1)|(self.iab12_df['gc']<0), 'gc'] = np.nan
        # ga = 0.1

    def fitting_gagc(self, show_graphs=True):
        self._adjusting_input_pm()

        # Comparing variables for IAB3 and IAB12
        # print(self.iab12_df[['es','ea','delta','Rn_Avg_MJmh','G_Avg_MJmh','air_density','VPD','psychrometric_cte','gc']].describe())
        # print(self.iab3_df[['es','e','delta','LE_MJmh','shf_Avg_MJmh','air_density','VPD_kPa','psychrometric_kPa','gc','ga']].describe())

        # self.iab12_df[['gc']].plot()
        self._gagc()

        ga = 0.05

        self.iab3_df_gagc = self.iab3_df.set_index('TIMESTAMP').resample('1d').mean()[['ga','gc']]
        # print(self.iab3_df_gagc)
        self.iab3_df_gagc.loc[self.iab3_df_gagc['gc']<0, 'gc'] = 0
        self.iab3_df_gagc.reset_index(inplace=True)

        # Calculo gc baseado no Rn (Maneira Errada)
        self.iab12_df['gc'] = (self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['psychrometric_cte']*ga)/(self.iab12_df['delta']*(self.iab12_df['Rn_Avg_MJmh']-self.iab12_df['G_Avg_MJmh'])+self.iab12_df['air_density']*3600*1.013*10**(-3)*self.iab12_df['VPD']*ga-self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['delta']-self.iab12_df['Rn_Avg_MJmh']*self.iab12_df['psychrometric_cte'])
        self.iab12_df.loc[(self.iab12_df['gc']>1)|(self.iab12_df['gc']<0), 'gc'] = np.nan

        # Adoção do uso do gc do iab3 para calculo do LE para depois calcular gc
        for i in range(1, 13, 1):
            self.iab12_df.loc[(self.iab12_df['TIMESTAMP'].dt.month==i), 'gc'] = self.iab3_df_gagc.loc[(self.iab3_df_gagc['TIMESTAMP'].dt.month==i)&(self.iab3_df_gagc['TIMESTAMP'].dt.year>2018), 'gc'].mean()

        self.iab12_df['LE_iab12'] = (self.iab12_df['delta']*self.iab12_df['Rn_Avg_MJmh']+3600*self.iab12_df['air_density']*1.013*10**(-3)*self.iab12_df['VPD']*ga)/((self.iab12_df['delta']+self.iab12_df['psychrometric_cte']*(1+ga/self.iab12_df['gc'])))
        self.iab12_df['ET_iab12'] = self.iab12_df['LE_iab12']/2.45

        self.iab12_df['gc_le'] = (self.iab12_df['LE_iab12']*self.iab12_df['psychrometric_cte']*ga)/(self.iab12_df['delta']*(self.iab12_df['LE_iab12']-self.iab12_df['G_Avg_MJmh'])+self.iab12_df['air_density']*3600*1.013*10**(-3)*self.iab12_df['VPD']*ga-self.iab12_df['LE_iab12']*self.iab12_df['delta']-self.iab12_df['LE_iab12']*self.iab12_df['psychrometric_cte'])
        self.iab12_df.loc[(self.iab12_df['gc']>1)|(self.iab12_df['gc']<0), 'gc'] = np.nan

        self.iab12_df.reset_index(inplace=True)

        self.iab12_df_gc = self.iab12_df.set_index('TIMESTAMP').resample('1d').mean()[['gc','gc_le']]
        self.iab12_df_gc.reset_index(inplace=True)


        print('Mean IAB3 ga: ',self.iab3_df_gagc['ga'].mean())

        meses = np.arange(1,13,1)
        gc_iab3 = []
        gc2 = []
        gc_iab2_le = []
        # ga = []

        for i in range(1, 13, 1):
            gc_iab3_monthly = self.iab3_df_gagc.loc[(self.iab3_df_gagc['TIMESTAMP'].dt.month==i)&(self.iab3_df_gagc['TIMESTAMP'].dt.year>2018), 'gc'].mean()
            # gc_iab12_monthly_Rn = self.iab12_df_gc.loc[(self.iab12_df_gc['TIMESTAMP'].dt.month==i)&(self.iab12_df_gc['TIMESTAMP'].dt.year>2018),'gc'].mean()
            gc_iab12_monthly_Le = self.iab12_df_gc.loc[(self.iab12_df_gc['TIMESTAMP'].dt.month==i)&(self.iab12_df_gc['TIMESTAMP'].dt.year>2018),'gc_le'].mean()

            # ga_iab3_monthly = self.iab3_df_gagc.loc[(self.iab3_df_gagc['TIMESTAMP'].dt.month==i)&(self.iab3_df_gagc['TIMESTAMP'].dt.year>2018), 'ga'].mean()

            gc_iab3.append(gc_iab3_monthly)
            gc_iab2_le.append(gc_iab12_monthly_Le)
            # ga.append(ga_iab3_monthly)

        gc_iab3 = np.array(gc_iab3)
        gc_iab2_le = np.array(gc_iab2_le)

        meses02 = meses[~np.isnan(gc_iab3)]
        gc_iab3 = gc_iab3[~np.isnan(gc_iab3)]

        meses02_gc2_le = meses[~np.isnan(gc_iab2_le)]
        gc_iab2_le = gc_iab2_le[~np.isnan(gc_iab2_le)]

        coefs_iab3 = np.poly1d(np.polyfit(meses02, gc_iab3, 2))
        coefs_iab2_le = np.poly1d(np.polyfit(meses02_gc2_le, gc_iab2_le, 2))

        print('Equação 2º grau fit IAB3:\n',coefs_iab3)
        print('Equação 2º grau fit IAB12:\n',coefs_iab2_le)

        if show_graphs:
            fig, ax = plt.subplots(3, figsize=(10,9))

            ax[1].scatter(self.iab3_df_gagc['TIMESTAMP'], self.iab3_df_gagc['gc'], color='blue')
            # ax[1].scatter(self.iab12_df_gc['TIMESTAMP'], self.iab12_df_gc['gc'], color='red')
            ax[1].scatter(self.iab12_df_gc['TIMESTAMP'], self.iab12_df_gc['gc_le'], color='purple')

            ax[1].set_yscale('log')
            ax[1].set_ylim((0.0001,0.1))
            ax[1].set_title('gc')

            ax[2].scatter(self.iab3_df_gagc['TIMESTAMP'], self.iab3_df_gagc['ga'], color='blue')
            ax[2].axhline(y=self.iab3_df_gagc['ga'].mean(), color='darkblue')
            ax[2].set_title('ga')
            ax[2].set_yscale('log')
            ax[2].set_ylim((0.001,1))

            for i in range(1, 13, 1):
                gc_iab3_monthly = self.iab3_df_gagc.loc[(self.iab3_df_gagc['TIMESTAMP'].dt.month==i)&(self.iab3_df_gagc['TIMESTAMP'].dt.year>2018), 'gc'].mean()
                # gc_iab12_monthly_Rn = self.iab12_df_gc.loc[(self.iab12_df_gc['TIMESTAMP'].dt.month==i)&(self.iab12_df_gc['TIMESTAMP'].dt.year>2018),'gc'].mean()
                gc_iab12_monthly_Le = self.iab12_df_gc.loc[(self.iab12_df_gc['TIMESTAMP'].dt.month==i)&(self.iab12_df_gc['TIMESTAMP'].dt.year>2018),'gc_le'].mean()

                ga_iab3_monthly = self.iab3_df_gagc.loc[(self.iab3_df_gagc['TIMESTAMP'].dt.month==i)&(self.iab3_df_gagc['TIMESTAMP'].dt.year>2018), 'ga'].mean()
                ax[0].scatter(i, gc_iab3_monthly, color='blue')
                # ax[0].scatter(i, gc_iab12_monthly_Rn, color='red')
                ax[0].scatter(i, gc_iab12_monthly_Le, color='purple')

            ax[0].set_yscale('log')
            ax[0].plot(meses, coefs_iab3(meses), color='blue', label='gc_iab3')
            ax[0].plot(meses, coefs_iab2_le(meses), color='purple', label='gc_iab2_le')
            ax[0].set_title('Monthly gc')
            ax[0].legend()

        self.iab12_df_gc_monthly = self.iab12_df_gc.set_index('TIMESTAMP').resample('1m').mean()
        self.iab12_df_gc_monthly.reset_index(inplace=True)
        self.iab3_df_gagc = self.iab3_df.set_index('TIMESTAMP').resample('1m').mean()[['ga','gc']]
        self.iab3_df_gagc.reset_index(inplace=True)

    @classmethod
    def MBE(self, y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values

        Returns:
            mbe (float): Biais score
        '''
        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        # y_true = y_true.reshape(len(y_true),1)
        # y_pred = y_pred.reshape(len(y_pred),1)
        diff = (y_true-y_pred)
        # print(len(y_true), len(y_pred))
        mbe = diff.mean()
        # print('MBE = ', mbe)
        return mbe

    def lstm_model_forecast(self, model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    def dnn_model(self, train_X, val_X, train_y, val_y, learning_rate, epochs, batch_size):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

        models = []

        for e in epochs:
            for l in learning_rate:
                optimizer = tf.keras.optimizers.SGD(lr=l)
                for b in batch_size:
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(1000, input_shape=[np.shape(train_X)[1]], activation='relu'),
                        # tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(800, activation='relu'),
                        tf.keras.layers.Dense(600, activation='relu'),
                        tf.keras.layers.Dense(150, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                        ])

                    model.compile(optimizer=optimizer,
                                  loss=tf.keras.losses.Huber(),
                                  metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])
                    history = model.fit(x=train_X, y=train_y,
                                        epochs=e, batch_size=b, verbose=0,
                                        validation_data=(val_X, val_y))
                    # last_mae_t = history.history['mae'][-1]
                    # last_mae_v = history.history['val_mae'][-1]
                    last_mae_v = history.history['val_mean_absolute_error'][-1]
                    last_rmse_v = history.history['val_root_mean_squared_error'][-1]
                    print('MAE:\t',history.history['val_mean_absolute_error'][-1])
                    print('RMSE:\t',history.history['val_root_mean_squared_error'][-1])

                    models.append(model)

                    # plt.title(f'Batch_size: {b} | LR: {l:.2f} | ')
                    # plt.plot(history.history['mean_absolute_error'], label='mae')
                    # plt.plot(history.history['val_mean_absolute_error'], label='Validation')
                    #
                    # plt.legend(loc='best')
                    # plt.xlabel('# Epochs')
                    # plt.ylabel('MAE')
                    # # plt.savefig(r'G:\Meu Drive\USP-SHS\Resultados_processados\Gapfilling\ANN\imgs\dnn\{}-epochs_{}-lr_{}-bs.png'.format(e,l,b))
                    # plt.show()
                    tf.keras.backend.clear_session()
                    tf.random.set_seed(51)
        return models

    def lstm_univariate_model(self, length, generator_train, generator_val, epochs=10):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

        model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=0, input_shape=(length, 1)),
            tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, dropout=0.4),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
        history = model.fit(generator_train, epochs=epochs,validation_data=generator_train)

        # plt.title('')
        # plt.plot(history.history['mae'], label='Training')
        # plt.plot(history.history['val_mae'], label='Validation')
        #
        # plt.legend(loc='best')
        # plt.xlabel('# Epochs')
        # plt.ylabel('MAE')
        # plt.show()

        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        return model

    def lstm_conv1d_univariate_model(self, length, generator_train, generator_val, epochs=10):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

        model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=0, input_shape=(length, 1)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=12, strides=1, activation='relu'),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        print(model.summary())
        model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
        # model.fit(generator_train, epochs=epochs, validation_split=0.2)
        model.fit_generator(generator_train, epochs=epochs, validation_data=generator_val)
        # model.evaluate_generator(generator_train)
        # print(model.predict_generator(generator_val))

        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

        return model

    def lstm_multivariate_model(self, length, generator_train, generator_val, epochs=10, n_columns=8):
        tf.keras.backend.clear_session()
        tf.random.set_seed(50)

        model = tf.keras.Sequential([
            # tf.keras.layers.Masking(mask_value=0, input_shape=(length, 8)),
            tf.keras.layers.LSTM(32, activation='relu', input_shape=(length, n_columns), return_sequences=True),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
        history = model.fit(generator_train, epochs=epochs, validation_data=generator_train, verbose=0)

        tf.keras.backend.clear_session()
        tf.random.set_seed(50)

        return model

    def fill_ET(self, listOfmethods):
        self.iab3_ET_timestamp = self.dropping_bad_data()
        self.ET_names = []

        if 'baseline' in listOfmethods:
            print('#####\t Baseline \t#####')
            self.ET_names.append('ET_baseline')

            iab3_df_copy = self.dropping_bad_data()
            iab3_df_copy.dropna(subset=['ET'], inplace=True)

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')
            df_date_range = pd.DataFrame({'TIMESTAMP':date_range})
            iab3_alldates = pd.merge(left=df_date_range, right=iab3_df_copy[['TIMESTAMP','ET']], on='TIMESTAMP', how='outer')

            # print(iab3_alldates)

            iab3_alldates['ET_baseline'] = iab3_alldates['ET'].fillna(method='ffill', limit=1)

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP', 'ET_baseline']], on='TIMESTAMP', how='outer')

        if 'mdv' in listOfmethods:
            print('#####\t MDV \t#####')
            n_days_list = [3,5, 7]

            iab3_df_copy = self.dropping_bad_data()
            iab3_df_copy.dropna(subset=['ET'], inplace=True)

            a, b = train_test_split(iab3_df_copy[['TIMESTAMP', 'ET']])

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')
            df_date_range = pd.DataFrame({'TIMESTAMP':date_range})

            iab3_alldates = pd.merge(left=df_date_range, right=a, on='TIMESTAMP', how='outer')

            b.rename(columns={"ET":'ET_val_mdv'}, inplace=True)

            iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')

            self.ET_names.append(f'ET_mdv_{n_days_list}')

            column_names = []
            for n in n_days_list:
                print(n)
                # self.ET_names.append(f'ET_mdv_{n}')
                column_names.append(f'ET_mdv_{n}')
                self._adjacent_days(df=iab3_alldates, n_days=n)

                # for i, row in iab3_alldates.loc[iab3_alldates['ET'].isna()].iterrows():
                for i, row in iab3_alldates.iterrows():
                    iab3_alldates.loc[i, f'ET_mdv_{n}'] = iab3_alldates.loc[(iab3_alldates['TIMESTAMP'].isin(row[f'timestamp_adj_{n}']))&
                                                                                 (iab3_alldates['ET'].notna()), 'ET'].mean()


            iab3_alldates[f'ET_mdv_{n_days_list}'] = iab3_alldates[f'ET_mdv_{n_days_list[0]}']

            for n in column_names:
                print(n)
                iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].isna())&
                                  (iab3_alldates[n].notna()), f'ET_mdv_{n_days_list}'] = iab3_alldates.loc[(iab3_alldates[n].notna())&
                                                                           (iab3_alldates[f'ET_mdv_{n_days_list}'].isna()), n]

                    # iab3_alldates.loc[i, f'ET_mdv_{n_days_list}'] = iab3_alldates.loc[(iab3_alldates['TIMESTAMP'])]
            # print(iab3_alldates[['TIMESTAMP',f'ET_mdv_{n_days_list}',f'ET_mdv_{n_days_list[0]}',f'ET_mdv_{n_days_list[1]}']])
            # print(iab3_alldates[['TIMESTAMP',f'ET_mdv_{n_days_list}',f'ET_mdv_{n_days_list[0]}',f'ET_mdv_{n_days_list[1]}']].describe())
            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP']+[f'ET_mdv_{n_days_list}']], on='TIMESTAMP', how='outer')

        if 'mdv2' in listOfmethods:
            n_days_list = [3,5,7]

            # self.ET_names.append(f'ET_mdv2_{n_days_list}')
            print('#####\t MDV2 \t#####')

            iab3_df_copy = self.dropping_bad_data()
            iab3_df_copy.dropna(subset=['ET'], inplace=True)

            a, b = train_test_split(iab3_df_copy[['TIMESTAMP', 'ET']])
            b.rename(columns={"ET":'ET_val_mdv'}, inplace=True)

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')
            df_date_range = pd.DataFrame({'TIMESTAMP':date_range})
            iab3_alldates = pd.merge(left=df_date_range, right=a, on='TIMESTAMP', how='outer')
            iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')
            # print(iab3_alldates.loc[['ET_val_mdv']])
            column_names = []
            for n in n_days_list:
                print(n)
                column_names.append(f'ET_mdv_{n}')
                self._adjacent_days(df=iab3_alldates, n_days=n)

                for i, row in iab3_alldates.iterrows():
                    iab3_alldates.loc[i, f'ET_mdv_{n}'] = iab3_alldates.loc[(iab3_alldates['TIMESTAMP'].isin(row[f'timestamp_adj_{n}']))&
                                                                                 (iab3_alldates['ET'].notna()), 'ET'].mean()
            iab3_alldates[f'ET_mdv_{n_days_list}'] = iab3_alldates[f'ET_mdv_{n_days_list[0]}']

            # print(iab3_alldates)
            for n in column_names:
                print(n)
                iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].isna())&
                                  (iab3_alldates[n].notna()), f'ET_mdv_{n_days_list}'] = iab3_alldates.loc[(iab3_alldates[n].notna())&
                                                                           (iab3_alldates[f'ET_mdv_{n_days_list}'].isna()), n]


            print('Validation metrics:')
            # print(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}']).notna()&(iab3_alldates['ET_val_mdv'].notna()), ['ET_val_mdv',f'ET_mdv_{n_days_list}']])
            print('MAE: \t',mean_absolute_error(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), 'ET_val_mdv'],
                  iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), f'ET_mdv_{n_days_list}']))
            print('RMSE: \t', mean_squared_error(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), 'ET_val_mdv'],
                  iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), f'ET_mdv_{n_days_list}'])**(1/2))
            print('MBE: \t', self.MBE(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), 'ET_val_mdv'].values,
                  iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), f'ET_mdv_{n_days_list}'].values))


            val_df = pd.DataFrame({'y_predict': iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), f'ET_mdv_{n_days_list}'].values,
                                   'val_y':iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), 'ET_val_mdv'].values,
                                  'TIMESTAMP':iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&(iab3_alldates['ET_val_mdv'].notna()), 'TIMESTAMP'].values})
            print(val_df)
            print(val_df[['y_predict','val_y']].corr()['y_predict'][1])

            print('MAE (daytime):',mean_absolute_error(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                                                          (iab3_alldates['ET_val_mdv'].notna())&
                                                                          (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                                                          (iab3_alldates['TIMESTAMP'].dt.hour<18), 'ET_val_mdv'],
                  iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                     (iab3_alldates['ET_val_mdv'].notna())&
                                     (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                     (iab3_alldates['TIMESTAMP'].dt.hour<18), f'ET_mdv_{n_days_list}']))
            print('RMSE (daytime): \t', mean_squared_error(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                                                   (iab3_alldates['ET_val_mdv'].notna())&
                                                                   (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                                                   (iab3_alldates['TIMESTAMP'].dt.hour<18), 'ET_val_mdv'],
                                                 iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                                                   (iab3_alldates['ET_val_mdv'].notna())&
                                                                   (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                                                   (iab3_alldates['TIMESTAMP'].dt.hour<18), f'ET_mdv_{n_days_list}'])**(1/2))
            print('MBE (daytime): \t', self.MBE(iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                                        (iab3_alldates['ET_val_mdv'].notna())&
                                                        (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                                        (iab3_alldates['TIMESTAMP'].dt.hour<18), 'ET_val_mdv'].values,
                                      iab3_alldates.loc[(iab3_alldates[f'ET_mdv_{n_days_list}'].notna())&
                                                        (iab3_alldates['ET_val_mdv'].notna())&
                                                        (iab3_alldates['TIMESTAMP'].dt.hour>=6)&
                                                        (iab3_alldates['TIMESTAMP'].dt.hour<18), f'ET_mdv_{n_days_list}'].values))
            print(val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                             (val_df['TIMESTAMP'].dt.hour<18), ['y_predict','val_y']].corr()['y_predict'][1])

            self.plot_corr_val(ypred=val_df['y_predict'], ytrue=val_df['val_y'], method='mdv', corr=val_df.corr()['y_predict'][1])
            self.plot_corr_val(ypred=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'y_predict'],
                               ytrue=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'val_y'],
                               method='mdv_daytime', corr=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), ['y_predict','val_y']].corr()['y_predict'][1])



        if 'lr' in listOfmethods:
            print('#####\t LR \t#####')
            self.ET_names.append('ET_lr')
            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
            column_x_ET = column_x + ['ET']
            iab3_df_copy_na = iab3_df_copy.copy()
            iab3_df_copy_na.dropna(subset=column_x_ET, inplace=True)

            X = iab3_df_copy_na[column_x+['TIMESTAMP']]
            y = iab3_df_copy_na[['ET','TIMESTAMP']]

            train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, shuffle=True)

            lm = LinearRegression()
            model = lm.fit(train_X[column_x], train_y['ET'])

            lm_prediction = model.predict(val_X[column_x])

            val_y['prediction'] = lm_prediction
            # print(lm_prediction)
            print('Validation metrics:')
            print('MAE: \t',mean_absolute_error(val_y['ET'], lm_prediction))
            print('RMSE: \t', mean_squared_error(val_y['ET'], lm_prediction)**(1/2))
            print('MBE: \t', self.MBE(val_y['ET'], lm_prediction))

            print('MAE (daytime): \t',mean_absolute_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))
            print('RMSE (daytime): \t', mean_squared_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction'])**(1/2))
            print('MBE (daytime): \t', self.MBE(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))

            # val_df = pd.DataFrame({'y_predict': val_y['prediction'].values, 'val_y':val_y['ET'].values, 'TIMESTAMP':val_y['TIMESTAMP'].values})
            val_df = val_y.copy()
            # print(val_df)
            print(val_y)
            print(val_df[['prediction','ET']].corr()['prediction'][1])
            print(val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])

            self.plot_corr_val(ypred=val_df['prediction'], ytrue=val_df['ET'], method='lr', corr=val_df[['prediction','ET']].corr()['prediction'][1])
            self.plot_corr_val(ypred=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'prediction'],
                               ytrue=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'ET'],
                               method='lr_daytime',
                               corr=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])



            print('Coef: \t', lm.coef_)
            print('Intercept:\t ', lm.intercept_)

            print('All data metrics:')
            iab3_df_copy.dropna(subset=column_x, inplace=True)
            lm2 = LinearRegression()
            model_alldata = lm2.fit(X[column_x], y['ET'])
            lm_fill = model_alldata.predict(iab3_df_copy[column_x])
            iab3_df_copy['ET_lr'] = lm_fill
            iab3_df_copy.loc[iab3_df_copy['ET_lr']<0, 'ET_lr'] = 0

            iab3_df_notna = iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_lr'].notna())]
            # print(iab3_df_copy)
            print('MAE: \t', mean_absolute_error(iab3_df_notna['ET'], iab3_df_notna['ET_lr']))
            print('RMSE: \t', mean_squared_error(iab3_df_notna['ET'], iab3_df_notna['ET_lr'])**(1/2))
            print('MBE: \t', self.MBE(iab3_df_notna['ET'], iab3_df_notna['ET_lr']))

            # print('MAE (daytime): \t', mean_absolute_error(iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                                  (iab3_df_notna['TIMESTAMP'].dt.hour<18),'ET'],
            #                                                iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                                  (iab3_df_notna['TIMESTAMP'].dt.hour<18),'ET_lr']))
            # print('RMSE (daytime): \t', mean_squared_error(iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                                  (iab3_df_notna['TIMESTAMP'].dt.hour<18),'ET'],
            #                                                iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                                  (iab3_df_notna['TIMESTAMP'].dt.hour<18), 'ET_lr'])**(1/2))
            # print('MBE (daytime): \t', self.MBE(iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                       (iab3_df_notna['TIMESTAMP'].dt.hour<18),'ET'],
            #                                     iab3_df_notna.loc[(iab3_df_notna['TIMESTAMP'].dt.hour>=6)&
            #                                                       (iab3_df_notna['TIMESTAMP'].dt.hour<18),'ET_lr']))

            # print(iab3_df_copy[['ET','ET_lr']])
            # print(iab3_df_notna.loc[(iab3_df_notna['ET'].notna())&(iab3_df_notna['ET_lr'].notna()),['ET', 'ET_lr']].describe())
            print('Coef: \t', lm2.coef_)
            print('Intercept:\t ', lm2.intercept_)
            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_df_copy[['TIMESTAMP', 'ET_lr']], on='TIMESTAMP', how='outer')

        if 'rfr' in listOfmethods:
            print('#####\t RFR \t#####')
            self.ET_names.append('ET_rfr')

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
            column_x_ET = column_x + ['ET'] + ['TIMESTAMP']
            iab3_df_copy_na = iab3_df_copy.copy()
            iab3_df_copy_na.dropna(subset=column_x_ET, inplace=True)

            X = iab3_df_copy_na[column_x+['TIMESTAMP']]
            y = iab3_df_copy_na[['ET','TIMESTAMP']]

            train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, shuffle=True)

            print('Validation metrics:')
            et_model_RFR = RandomForestRegressor(random_state=1, criterion='mae')
            et_model_RFR.fit(train_X[column_x], train_y['ET'])
            rfr_validation = et_model_RFR.predict(val_X[column_x])
            val_y['prediction'] = rfr_validation

            print('MAE: \t',mean_absolute_error(val_y['ET'], rfr_validation))
            print('RMSE: \t', mean_squared_error(val_y['ET'], rfr_validation)**(1/2))
            print('MBE: \t', self.MBE(val_y['ET'], rfr_validation))


            print('MAE (daytime): \t',mean_absolute_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))
            print('RMSE (daytime): \t', mean_squared_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction'])**(1/2))
            print('MBE (daytime): \t', self.MBE(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))

            # val_df = pd.DataFrame({'y_predict': rfr_validation, 'val_y':val_y.values})
            val_df = val_y.copy()
            print(val_df)
            print(val_df[['prediction', 'ET']].corr()['prediction'][1])
            print(val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])

            self.plot_corr_val(ypred=val_df['prediction'], ytrue=val_df['ET'], method='rfr', corr=val_df.corr()['prediction'][1])
            self.plot_corr_val(ypred=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'prediction'],
                               ytrue=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'ET'],
                               method='rfr_daytime',
                               corr=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])




            print('Feature importance:\t ', et_model_RFR.feature_importances_)

            print('All data metrics:')




            # iab3_df_copy_na['ET_rfr'] = rfr_alldata
            iab3_df_copy.dropna(subset=column_x, inplace=True)
            et_model_RFR_all = RandomForestRegressor(random_state=1, criterion='mae')
            et_model_RFR_all.fit(iab3_df_copy_na[column_x], iab3_df_copy_na['ET'])
            rfr_alldata = et_model_RFR_all.predict(iab3_df_copy[column_x])
            # rfr_prediction = et_model_RFR.predict(iab3_df_copy[column_x])
            iab3_df_copy['ET_rfr'] = rfr_alldata
            iab3_df_copy.loc[iab3_df_copy['ET_rfr']<0, 'ET_rfr'] = 0
            print('MAE: \t', mean_absolute_error(iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET'], iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET_rfr']))
            print('RMSE: \t', mean_squared_error(iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET'], iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET_rfr'])**(1/2))
            print('MBE: \t', self.MBE(iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET'], iab3_df_copy.loc[(iab3_df_copy['ET'].notna())&(iab3_df_copy['ET_rfr'].notna()),'ET_rfr']))

            print('Feature importance:\t ', et_model_RFR_all.feature_importances_)

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_df_copy[['TIMESTAMP', 'ET_rfr']], on='TIMESTAMP', how='outer')

        if 'pm' in listOfmethods:
            print('#####\t PM \t#####')
            self.ET_names.append('ET_pm')

            # self._apply
            # self.d
            self._adjusting_input_pm()
            self._gagc()

            self.fitting_gagc(show_graphs=False)

            print(self.iab12_df.loc[self.iab12_df['TIMESTAMP'].dt.year>2018,'ET_iab12'].describe())

            pm_inputs_iab3 = ['delta', 'Rn_Avg_MJmh', 'shf_Avg_MJmh', 'air_density', 'VPD_kPa', 'ga','LE_MJmh','psychrometric_kPa', 'gc', 'TIMESTAMP']
            pm_inputs_iab3_ET = pm_inputs_iab3 + ['ET']

            iab3_df_copy = self.dropping_bad_data()
            iab3_df_copy.dropna(subset=pm_inputs_iab3, inplace=True)
            for i, row in self.iab3_df_gagc.iterrows():
                # print(row[['ga','gc','TIMESTAMP']])
                iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(iab3_df_copy['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'ga_mes'] = row['ga']
                iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(iab3_df_copy['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'gc_mes'] = row['gc']
                # &(iab3_df_copy['TIMESTAMP'].dt.year==row['TIMESTAMP'].year)
            for i, row in self.iab12_df_gc_monthly.iterrows():
                # print(row[['gc_le','TIMESTAMP']])

                iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(iab3_df_copy['TIMESTAMP'].dt.year==row['TIMESTAMP'].year),'gc_mes_iab2_le'] = row['gc_le']

            # print(iab3_df_copy[['TIMESTAMP', 'gc_mes_iab2_le']])


            ##!!!!!! Talvez colcular o ET com os inputs do iab12 e deixar o gc_le para comparação dos iab12 com iab3
            ## Para alcançar o gc_le (iab12) é preciso utilizar os dados do iab3, para gerar a sazonildade da variavel e ai sim utilizar o ET com os dados das outras estações
            # print(self.iab12_df)
            iab3_df_copy = pd.merge(left=iab3_df_copy, right=self.iab12_df.loc[self.iab12_df['TIMESTAMP'].dt.year>2018,['TIMESTAMP','ET_iab12']], on='TIMESTAMP', how='outer')


            # print(iab3_df_copy[['TIMESTAMP','ga_mes','gc_mes']])

            iab3_df_copy['ET_pm'] = (iab3_df_copy['delta']*(iab3_df_copy['Rn_Avg_MJmh']-iab3_df_copy['shf_Avg_MJmh'])+3600*iab3_df_copy['air_density']*1.013*10**(-3)*iab3_df_copy['VPD_kPa']*iab3_df_copy['ga_mes'])/(2.45*(iab3_df_copy['delta']+iab3_df_copy['psychrometric_kPa']*(1+iab3_df_copy['ga_mes']/iab3_df_copy['gc_mes'])))
            # print(self.iab3_df_gagc[['TIMESTAMP','ga', 'gc']])
            # print(iab3_df_copy.loc[iab3_df_copy['ET_pm']>0])
            # print(iab3_df_copy[pm_inputs_iab3+['ET_pm']].describe())
            # print(iab3_df_copy[['ET_pm', 'ET_iab12']].describe())
            # print(iab3_df_copy.loc[(iab3_df_copy['ET_pm'].isna())&(iab3_df_copy['TIMESTAMP'].dt.year==2019),['TIMESTAMP','ET_pm','ET_iab12']])
            iab3_df_copy.loc[iab3_df_copy['ET_pm'].isna(), 'ET_pm'] = iab3_df_copy['ET_iab12']
            print(iab3_df_copy.loc[(iab3_df_copy['TIMESTAMP'].dt.year==2020),'ET_pm'].sum())

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_df_copy[['TIMESTAMP', 'ET_pm']], on='TIMESTAMP', how='outer')
            self.iab3_ET_timestamp['ET_pm'] = self.iab3_ET_timestamp['ET_pm'].astype(float)

        if 'pm2' in listOfmethods:
            # self._gagc_2()
            self._gagc()
            pm_inputs_iab3 = ['delta', 'Rn_Avg_MJmh', 'shf_Avg_MJmh', 'air_density', 'VPD_kPa', 'ga','LE_MJmh','psychrometric_kPa', 'gc', 'TIMESTAMP']
            pm_inputs_iab3_ET = pm_inputs_iab3 + ['ET']
            iab3_df_copy = self.dropping_bad_data()
            iab3_df_copy.dropna(subset=pm_inputs_iab3_ET, inplace=True)

            train, val = train_test_split(iab3_df_copy, shuffle=True, random_state=1)
            train = train.copy()
            val = val.copy()

            self.iab3_df_gagc_2 = train.set_index('TIMESTAMP').resample('1m').mean()[['ga','gc']]
            self.iab3_df_gagc_2.reset_index(inplace=True)

            # print(self.iab3_df_gagc_2)
            for i, row in self.iab3_df_gagc_2.iterrows():
                val.loc[(val['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(val['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'ga_mes'] = row['ga']
                val.loc[(val['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(val['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'gc_mes'] = row['gc']

            # print(val[['ga_mes','gc_mes']])

            val['ET_pm'] = (val['delta']*(val['Rn_Avg_MJmh']-val['shf_Avg_MJmh'])+3600*val['air_density']*1.013*10**(-3)*val['VPD_kPa']*val['ga_mes'])/(2.45*(val['delta']+val['psychrometric_kPa']*(1+val['ga_mes']/val['gc_mes'])))

            val.dropna(subset=pm_inputs_iab3_ET+['ga_mes','gc_mes','ET_pm'], inplace=True)

            print(val[['ET_pm','ET']])
            # print(val[['ET_pm', 'ET']].corr())
            corr = pd.DataFrame({'real':val['ET'].values, 'predict':val['ET_pm'].astype(float).values, 'TIMESTAMP':val['TIMESTAMP'].values})
            print(corr[['real','predict']].corr())
            # print(corr.dtypes)

            print('MAE: \t', mean_absolute_error(val['ET'].values, val['ET_pm'].values))
            print('RMSE: \t', mean_squared_error(val['ET'].values, val['ET_pm'].values)**(1/2))
            print('MBE: \t', self.MBE(val['ET'].values, val['ET_pm'].values))

            print('MAE (daytime): \t', mean_absolute_error(val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                                   (val['TIMESTAMP'].dt.hour<18),'ET'].values,
                                                           val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                                   (val['TIMESTAMP'].dt.hour<18),'ET_pm'].values))
            print('RMSE (daytime): \t', mean_squared_error(val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                                   (val['TIMESTAMP'].dt.hour<18),'ET'].values,
                                                           val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                                   (val['TIMESTAMP'].dt.hour<18),'ET_pm'].values)**(1/2))
            print('MBE (daytime): \t', self.MBE(val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                        (val['TIMESTAMP'].dt.hour<18),'ET'].values,
                                                val.loc[(val['TIMESTAMP'].dt.hour>=6)&
                                                        (val['TIMESTAMP'].dt.hour<18),'ET_pm'].values))

            #
            # plt.scatter(val['ET'], val['ET_pm'])
            self.plot_corr_val(ytrue=corr['real'], ypred=corr['predict'], method='pm', corr=corr[['real','predict']].corr()['real'][1])
            self.plot_corr_val(ytrue=corr.loc[(corr['TIMESTAMP'].dt.hour>=6)&
                                              (corr['TIMESTAMP'].dt.hour<18),'real'],
                               ypred=corr.loc[(corr['TIMESTAMP'].dt.hour>=6)&
                                              (corr['TIMESTAMP'].dt.hour<18),'predict'],
                               method='pm_daytime',
                               corr=corr.loc[(corr['TIMESTAMP'].dt.hour>=6)&(corr['TIMESTAMP'].dt.hour<18),['real','predict']].corr()['real'][1])





        if 'dnn' in listOfmethods:
            print('#####\t DNN \t#####')
            self.ET_names.append('ET_dnn')

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
            column_x_ET = column_x + ['ET']
            iab3_df_copy.dropna(subset=column_x, inplace=True)
            iab3_df_copy_na = iab3_df_copy.copy()
            iab3_df_copy_na.dropna(subset=['ET'], inplace=True)

            X = iab3_df_copy_na[column_x+['TIMESTAMP']]
            y = iab3_df_copy_na[['ET','TIMESTAMP']]

            X_scale = preprocessing.scale(X[column_x])
            # print(X_scale)
            df_x_scale = pd.DataFrame(X_scale, columns=column_x)
            # print(df_x_scale)
            df_x_scale['TIMESTAMP'] = X['TIMESTAMP']

            # X_scale['TIMESTAMP'] = X['TIMESTAMP']
            train_X, val_X, train_y, val_y = train_test_split(df_x_scale, y, random_state=1, shuffle=True)

            print('Validation metrics:')
            model_val = self.dnn_model(train_X=train_X[column_x],val_X=val_X[column_x],
                                    train_y=train_y['ET'], val_y=val_y['ET'],
                                    learning_rate=[1e-2], epochs=[200], batch_size=[512])
            y_mbe = model_val[0].predict(val_X[column_x])
            print('MBE: \t', self.MBE(val_y['ET'].values, y_mbe))

            val_y['prediction'] = y_mbe


            print('MAE (daytime): \t',mean_absolute_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))
            print('RMSE (daytime): \t', mean_squared_error(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction'])**(1/2))
            print('MBE (daytime): \t', self.MBE(val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'ET'], val_y.loc[(val_y['TIMESTAMP'].dt.hour>=6)&(val_y['TIMESTAMP'].dt.hour<18),'prediction']))


            # print(y_mbe)
            # print(np.shape(y_mbe), np.shape(val_y.to_numpy()))
            # print(val_y.values)
            val_df = val_y.copy()
            # val_df = pd.DataFrame({'y_predict': y_mbe.reshape(len(y_mbe)), 'val_y':val_y['ET'].values})
            print(val_df)
            print(val_df[['prediction','ET']].corr()['prediction'][1])
            print(val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])

            self.plot_corr_val(ypred=val_df['prediction'], ytrue=val_df['ET'], method='dnn', corr=val_df.corr()['prediction'][1])
            self.plot_corr_val(ypred=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'prediction'],
                               ytrue=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'ET'],
                               method='dnn_daytime',
                               corr=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18), ['prediction', 'ET']].corr()['prediction'][1])

            print('All data metrics: ')
            models = self.dnn_model(train_X=X_scale,val_X=X_scale,
                                    train_y=y['ET'], val_y=y['ET'],
                                    learning_rate=[1e-2], epochs=[200], batch_size=[512])
            #
            X_predict = iab3_df_copy[column_x]
            X_predict = preprocessing.scale(X_predict)

            y_predict = models[0].predict(X_predict)

            iab3_df_copy['ET_dnn'] = y_predict
            iab3_df_copy.loc[iab3_df_copy['ET_dnn']<0, 'ET_dnn'] = 0

            # print(iab3_df_copy['ET_dnn'].describe())
            # print(models[0].predict(X_predict))
            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_df_copy[['TIMESTAMP', 'ET_dnn']], on='TIMESTAMP', how='outer')

        if 'lstm_u' in listOfmethods:
            print('LSTM_u...')
            self.ET_names.append('ET_lstm_u')

            length = 24
            batch_size = 12

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']

            iab3_df_copy.dropna(subset=column_x, inplace=True)

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')


            df_date_range = pd.DataFrame({'TIMESTAMP': date_range})

            iab3_alldates = pd.merge(left=df_date_range, right=iab3_df_copy, how='outer')
            iab3_alldates.loc[iab3_alldates['ET'].isnull(), "ET"] = 0

            train_X, val_X = train_test_split(iab3_alldates['ET'], shuffle=False)

            generator_train = TimeseriesGenerator(train_X, train_X, length=length, batch_size=batch_size)
            generator_val = TimeseriesGenerator(val_X, val_X, length=length, batch_size=batch_size)
            model = self.lstm_univariate_model(length=length,
                                               generator_train=generator_train,
                                               generator_val=generator_val,
                                               epochs=2)
            lstm_forecast = self.lstm_model_forecast(model, iab3_alldates['ET'].to_numpy()[..., np.newaxis], length)
            lstm_forecast = np.insert(lstm_forecast, 0, [0 for i in range(length-1)])

            validation_data = pd.DataFrame({'TIMESTAMP': date_range})
            validation_data['ET_lstm_u'] = lstm_forecast

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=validation_data[['TIMESTAMP','ET_lstm_u']], on='TIMESTAMP', how='outer')

            # print(self.iab3_ET_timestamp['ET_lstm_u'].describe())

        if 'lstm_u_v2' in listOfmethods:
            print('LSTM_u_v2')
            self.ET_names.append('ET_lstm_u_v2')

            length = 24
            batch_size = 128

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']

            iab3_alldates = iab3_df_copy.copy()
            iab3_alldates.loc[iab3_alldates['ET'].isnull(), "ET"] = 0
            print(iab3_alldates[['TIMESTAMP', 'ET']])

            train_X, val_X = train_test_split(iab3_alldates['ET'], shuffle=False)

            generator_all = TimeseriesGenerator(iab3_alldates['ET'], iab3_alldates['ET'], length=length,batch_size=batch_size)
            generator_train = TimeseriesGenerator(train_X, train_X, length=length, batch_size=batch_size)
            generator_val = TimeseriesGenerator(val_X, val_X, length=length, batch_size=batch_size)
            print('Split data (Training/Validation):')
            model = self.lstm_univariate_model(length=length,
                                               generator_train=generator_train,
                                               generator_val=generator_val,
                                               epochs=2)
            print('All data: ')
            model_alldata = self.lstm_univariate_model(length=length,
                                                       generator_train=generator_all,
                                                       generator_val=generator_all,
                                                       epochs=2)
            lstm_forecast = self.lstm_model_forecast(model_alldata, iab3_alldates['ET'].to_numpy()[..., np.newaxis], length)
            lstm_forecast = np.insert(lstm_forecast, 0, [0 for i in range(length-1)])

            # validation_data = pd.DataFrame({'TIMESTAMP': date_range})
            iab3_alldates['ET_lstm_u_v2'] = lstm_forecast

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP','ET_lstm_u_v2']], on='TIMESTAMP', how='outer')

        if 'lstm_conv1d_u' in listOfmethods:
            print('LSTM_Conv1D_u...')
            self.ET_names.append('ET_lstm_conv1d_u')

            length = 12
            batch_size = 128

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']

            iab3_df_copy.dropna(subset=column_x, inplace=True)

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')


            df_date_range = pd.DataFrame({'TIMESTAMP': date_range})

            iab3_alldates = pd.merge(left=df_date_range, right=iab3_df_copy, how='outer')
            iab3_alldates.loc[iab3_alldates['ET'].isnull(), "ET"] = 0

            train_X, val_X = train_test_split(iab3_alldates['ET'], shuffle=False)

            generator_train = TimeseriesGenerator(train_X, train_X, length=length, batch_size=batch_size)
            generator_val = TimeseriesGenerator(val_X, val_X, length=length, batch_size=batch_size)

            model = self.lstm_conv1d_univariate_model(length=length,
                                                      generator_train=generator_train,
                                                      generator_val=generator_val,
                                                      epochs=2)
            lstm_conv1d_forecast = self.lstm_model_forecast(model, iab3_alldates['ET'].to_numpy()[..., np.newaxis], length)
            lstm_conv1d_forecast = np.insert(lstm_conv1d_forecast, 0, [0 for i in range(length-1)])

            print(len(lstm_conv1d_forecast))
            validation_data = pd.DataFrame({'TIMESTAMP': date_range})
            validation_data['ET_lstm_conv1d_u'] = lstm_conv1d_forecast

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=validation_data[['TIMESTAMP','ET_lstm_conv1d_u']], on='TIMESTAMP', how='outer')

        if 'lstm_conv1d_u_v2' in listOfmethods:
            print('LSTM_Conv1D_u_v2...')
            self.ET_names.append('ET_lstm_conv1d_u_v2')
            length = 12
            batch_size = 128

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']

            iab3_alldates = iab3_df_copy.copy()
            iab3_alldates.loc[iab3_alldates['ET'].isnull(), "ET"] = 0

            train_X, val_X = train_test_split(iab3_alldates['ET'], shuffle=False)

            generator_train = TimeseriesGenerator(train_X, train_X, length=length, batch_size=batch_size)
            generator_val = TimeseriesGenerator(val_X, val_X, length=length, batch_size=1)

            model = self.lstm_conv1d_univariate_model(length=length,
                                                      generator_train=generator_train,
                                                      generator_val=generator_val,
                                                      epochs=2)
            lstm_conv1d_forecast = self.lstm_model_forecast(model, iab3_alldates['ET'].to_numpy()[..., np.newaxis], length)
            lstm_conv1d_forecast = np.insert(lstm_conv1d_forecast, 0, [0 for i in range(length-1)])

            # validation_data = pd.DataFrame({'TIMESTAMP': date_range})
            iab3_alldates['ET_lstm_conv1d_u_v2'] = lstm_conv1d_forecast

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP','ET_lstm_conv1d_u_v2']], on='TIMESTAMP', how='outer')

        if 'lstm_m' in listOfmethods:
            print('LSTM_m...')
            self.ET_names.append('ET_lstm_m')

            length = 24
            batch_size = 64

            iab3_df_copy = self.dropping_bad_data()
            column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'shf_Avg(1)','shf_Avg(2)','e','wind_speed']

            iab3_df_copy.dropna(subset=column_x, inplace=True)

            date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                       end=iab3_df_copy['TIMESTAMP'].max(),
                                       freq='30min')


            df_date_range = pd.DataFrame({'TIMESTAMP': date_range})

            iab3_alldates = pd.merge(left=df_date_range, right=iab3_df_copy, how='outer')

            column_x_n = ['Rn_Avg_n', 'RH_n','VPD_n','air_temperature_n','shf_Avg(1)_n','shf_Avg(2)_n','e_n','wind_speed_n']
            for i in column_x:
                iab3_alldates.loc[iab3_alldates[i].isna(), i] = 0
                iab3_alldates[f'{i}_n'] = preprocessing.scale(iab3_alldates[i])

            # print(iab3_alldates[column_x_n].describe())

            iab3_alldates['ET_shift'] = iab3_alldates['ET'].shift(1)
            generator_t = TimeseriesGenerator(iab3_alldates[column_x_n].to_numpy(), iab3_alldates['ET_shift'].to_numpy(), length=length, batch_size=3, shuffle=False)

            model = self.lstm_multivariate_model(length=length,
                                                 generator_train=generator_t,
                                                 generator_val=generator_t,
                                                 epochs=2)
            generator_prediction = TimeseriesGenerator(iab3_alldates[column_x_n].to_numpy(), iab3_alldates['ET_shift'].to_numpy(), length=length, batch_size=len(iab3_alldates[column_x_n]), shuffle=False)

            for i in generator_prediction:
                forecast = model.predict(i[0])
            lstm_forecast = np.insert(forecast, 0, [0 for i in range(length)])

            iab3_alldates['ET_lstm_multi_shift'] = lstm_forecast
            iab3_alldates['ET_lstm_m'] = iab3_alldates['ET_lstm_multi_shift'].shift(-1)
            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP','ET_lstm_m']], on='TIMESTAMP', how='outer')

        if 'lstm_m_v2' in listOfmethods:
            print('#####\t LSTM_M_v2 \t#####')
            self.ET_names.append('ET_lstm_m_v2')

            length = 8
            batch_size = 128

            iab3_df_copy = self.dropping_bad_data()
            column_x = [
                        'Rn_Avg',
                        'RH',
                        'VPD',
                        'air_temperature',
                        'shf_Avg(1)',
                        'shf_Avg(2)',
                        'e',
                        'wind_speed'
                        ]

            iab3_alldates = iab3_df_copy.copy()
            # print(iab3_df_copy[column_x+[]].describe())
            # a = iab3_df_copy.sort_values(by='TIMESTAMP')
            # print(a.loc[a[column_x[0]].notna(), 'TIMESTAMP'].diff().value_counts())

            column_x_n = [
                          'Rn_Avg_n',
                          'RH_n',
                          'VPD_n',
                          'air_temperature_n',
                          'shf_Avg(1)_n',
                          'shf_Avg(2)_n',
                          'e_n',
                          'wind_speed_n'
                          ]
            for i in column_x:
                iab3_alldates.loc[iab3_alldates[i].isna(), i] = 0
                iab3_alldates[f'{i}_n'] = preprocessing.scale(iab3_alldates[i])

            # print(iab3_alldates[['TIMESTAMP']+column_x])
            # print(iab3_alldates[column_x_n].describe())
            X_scale = preprocessing.scale(iab3_alldates[column_x])
            # print(X_scale)
            # print(iab3_alldates[column_x_n].to_numpy())

            iab3_alldates['ET_shift'] = iab3_alldates['ET'].shift(1)
            train_X, val_X, train_y, val_y = train_test_split(iab3_alldates[column_x_n], iab3_alldates[['ET_shift','TIMESTAMP']], shuffle=False, test_size=0.25)

            # print('train')
            # print(train_y)
            #
            # print('val')
            # print(val_y)
            # print(iab3_alldates['ET_shift'].to_numpy())
            # iab3_alldates.loc[iab3_alldates['ET_shift'].isna(),'ET_shift'] = 0
            generator_t = TimeseriesGenerator(train_X.to_numpy(),
                                              # iab3_alldates[column_x].to_numpy(),
                                              # iab3_alldates['ET_shift'].to_numpy(),
                                              train_y['ET_shift'].to_numpy(),
                                               length=length,
                                               batch_size=batch_size, shuffle=False)
            generator_val = TimeseriesGenerator(val_X.to_numpy(), val_y['ET_shift'].to_numpy(), length=length, batch_size=batch_size, shuffle=False)
            generator_all = TimeseriesGenerator(iab3_alldates[column_x_n].to_numpy(), iab3_alldates['ET_shift'].to_numpy(), length=length, batch_size=batch_size, shuffle=False)


            model = self.lstm_multivariate_model(length=length,
                                                 generator_train=generator_t,
                                                 generator_val=generator_val,
                                                 n_columns=len(column_x_n),
                                                 epochs=20)


            generator_val2 = TimeseriesGenerator(val_X.to_numpy(), val_y['ET_shift'].to_numpy(), length=length, batch_size=len(val_y['ET_shift']), shuffle=False)
            for i in generator_val2:
                forecast = model.predict(i[0])
            lstm_forecast_val = np.insert(forecast, 0, [0 for i in range(length)])
            print('Validation metrics:')
            df_val = pd.DataFrame({'ET_shift':val_y['ET_shift'], 'lstm_forecast_val':lstm_forecast_val, 'TIMESTAMP':val_y['TIMESTAMP']})
            df_val_notna = df_val.loc[(df_val['ET_shift'].notna())&(df_val['lstm_forecast_val'].notna())]
            print('MAE: \t',mean_absolute_error(df_val_notna['ET_shift'], df_val_notna['lstm_forecast_val']))
            print('RMSE: \t',mean_squared_error(df_val_notna['ET_shift'], df_val_notna['lstm_forecast_val'])**(1/2))
            print('MBE: \t', self.MBE(df_val_notna['ET_shift'], df_val_notna['lstm_forecast_val']))

            print('MAE (daytime): \t',mean_absolute_error(df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                                 (df_val_notna['TIMESTAMP'].dt.hour<18),'ET_shift'],
                                                df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                                 (df_val_notna['TIMESTAMP'].dt.hour<18),'lstm_forecast_val']))
            print('RMSE (daytime): \t',mean_squared_error(df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                                 (df_val_notna['TIMESTAMP'].dt.hour<18),'ET_shift'],
                                                df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                                 (df_val_notna['TIMESTAMP'].dt.hour<18),'lstm_forecast_val'])**(1/2))
            print('MBE (daytime): \t', self.MBE(df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                       (df_val_notna['TIMESTAMP'].dt.hour<18),'ET_shift'],
                                      df_val_notna.loc[(df_val_notna['TIMESTAMP'].dt.hour>=6)&
                                                       (df_val_notna['TIMESTAMP'].dt.hour<18),'lstm_forecast_val']))

            val_df = pd.DataFrame({'y_predict': df_val_notna['ET_shift'].values, 'val_y':df_val_notna['lstm_forecast_val'].values, 'TIMESTAMP':df_val_notna['TIMESTAMP'].values})

            print(val_df)
            print(val_df[['y_predict','val_y']].corr()['y_predict'][1])

            print(val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18),['y_predict','val_y']].corr()['y_predict'][1])

            self.plot_corr_val(ypred=val_df['y_predict'], ytrue=val_df['val_y'], method='lstm', corr=val_df.corr()['y_predict'][1])

            self.plot_corr_val(ypred=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'y_predict'],
                               ytrue=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&
                                                (val_df['TIMESTAMP'].dt.hour<18), 'val_y'],
                               method='lstm_daytime',
                               corr=val_df.loc[(val_df['TIMESTAMP'].dt.hour>=6)&(val_df['TIMESTAMP'].dt.hour<18),['y_predict','val_y']].corr()['y_predict'][1])



            generator_prediction = TimeseriesGenerator(iab3_alldates[column_x_n].to_numpy(), iab3_alldates['ET_shift'].to_numpy(), length=length, batch_size=len(iab3_alldates[column_x_n]), shuffle=False)
            print('All data metrics:')

            model_alldata = self.lstm_multivariate_model(length=length,
                                                 generator_train=generator_all,
                                                 generator_val=generator_all,
                                                 n_columns=len(column_x_n),
                                                 epochs=10)
            for i in generator_prediction:
                forecast = model_alldata.predict(i[0])
            lstm_forecast = np.insert(forecast, 0, [0 for i in range(length)])

            iab3_alldates['ET_lstm_multi_shift'] = lstm_forecast
            iab3_alldates.loc[iab3_alldates['ET_lstm_multi_shift']<0, 'ET_lstm_multi_shift'] = 0
            df_all_notna = pd.DataFrame({'ET_shift':iab3_alldates['ET_shift'], 'lstm_forecast_all':lstm_forecast})
            df_all_notna = df_all_notna.loc[(df_all_notna['ET_shift'].notna())&(df_all_notna['lstm_forecast_all'].notna())]
            print('MAE: \t', mean_absolute_error(df_all_notna['ET_shift'], df_all_notna['lstm_forecast_all']))
            print('RMSE: \t', mean_squared_error(df_all_notna['ET_shift'], df_all_notna['lstm_forecast_all'])**(1/2))


            iab3_alldates['ET_lstm_m_v2'] = iab3_alldates['ET_lstm_multi_shift'].shift(-1)

            self.iab3_ET_timestamp = pd.merge(left=self.iab3_ET_timestamp, right=iab3_alldates[['TIMESTAMP','ET_lstm_m_v2']], on='TIMESTAMP', how='outer')


        for i in self.ET_names:
            self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[i]<0, i] = 0
            # print(self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna())&(self.iab3_ET_timestamp[i].notna()), ['ET']+[i]].describe())

        # print(self.iab3_ET_timestamp[self.ET_names+['ET']].describe())

    def join_ET(self):
        self.filled_ET = []
        for et in self.ET_names:
            self.filled_ET.append(f'{et}_and_ET')
            self.iab3_ET_timestamp[f'{et}_and_ET'] = self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp['ET'].notna(), 'ET']
            self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[et]<0, et] = 0
            self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[f'{et}_and_ET'].isna(), f'{et}_and_ET'] = self.iab3_ET_timestamp[et]

        # print(self.iab3_ET_timestamp['ET_pm'].dtypes)
        # print(self.iab3_ET_timestamp['ET_pm'])
        # print(self.iab3_ET_timestamp['ET_pm'].astype(float))

        # print(self.iab3_ET_timestamp[self.iab3_ET_timestamp['ET_pm'].apply(lambda x: isinstance(x, np.nan))])

    def plot(self):
        print(self.iab3_ET_timestamp[self.ET_names+['ET']].describe())
        print(self.iab3_ET_timestamp[self.filled_ET+['ET']].cumsum().plot())
        # print(self.iab3_ET_timestamp[self.ET_names+['ET']].cumsum().plot())

    def plot_corr_val(self, ytrue, ypred, method, corr):
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots(1, figsize=(4,4), dpi=300)

        ax.scatter(ytrue, ypred,alpha=0.2)
        ax.plot([0,1],[0,1], color='red')
        # ax.set_title(f'{method}')
        ax.text(0.05,0.9,s=f'$\\rho$ = {corr:.3f}')
        ax.grid()

        # fig.show()
        ax.set_ylim((0,1))
        ax.set_xlim((0,1))
        # plt.show()
        fig.savefig(f'val_corr_{method}.png')

    def stats_others(self, stats=[], which=[]):
        iab3_df = self.iab3_df
        # fig_01, ax = plt.subplots(2, figsize=(10,3*len(stats)))

        if 'gaps_iab3' in stats:
            without_bigGAP = True

            import matplotlib.dates as md
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"

            columns = ['Rn_Avg', 'RH', 'VPD','shf_Avg(1)','shf_Avg(2)']
            fig_00, ax0 = plt.subplots(1, figsize=(10,4), dpi=300)


            b = iab3_df.set_index('TIMESTAMP')
            b.resample('D').count()[columns[0]].plot(ax=ax0, linestyle='dashdot',label='Radiação líquida')
            b.resample('D').count()[columns[1]].plot(ax=ax0, linestyle='solid',label='Umidade relativa')
            b.resample('D').count()[columns[2]].plot(ax=ax0, linestyle='dashed',label='Deficit de vapor de pressão')
            b.resample('D').count()[columns[3]].plot(ax=ax0, linestyle='dotted',label='Fluxo de calor do solo')
            # b.resample('D').count()[columns[4]].plot(ax=ax0, linestyle='dotted', label='Fluxo de calor do solo (sensor 2)')

            # ax0.set_title('Gaps in inputs variables (IAB3)')
            ax0.set_ylabel('Quantidade de dados diário')
            ax0.set_xlabel('')

            ax0.set_ylim((-1,50))
            ax0.legend(loc='lower left')
            plt.rc('axes', labelsize=14)
            # fig_00.savefig('iab3_gaps_meteo.png')



            fig_01, ax = plt.subplots(len(columns), figsize=(10,4*len(columns)+4))

            for j, variable in enumerate(columns):
                sorted_timestamp = self.iab3_ET_timestamp.sort_values(by='TIMESTAMP')
                variable_diff = sorted_timestamp.loc[(sorted_timestamp[f'{variable}'].notna()), 'TIMESTAMP'].diff()
                if without_bigGAP == False:
                    gaps_variable_index = variable_diff.loc[variable_diff>pd.Timedelta('00:30:00')].value_counts().sort_index().index
                    gap_cumulative = gaps_variable_index/pd.Timedelta('00:30:00')-1
                    gaps_variable_index = [str(i) for i in gaps_variable_index]
                    gaps_variable_count = variable_diff.loc[variable_diff>pd.Timedelta('00:30:00')].value_counts().sort_index().values

                elif without_bigGAP == True:
                    gaps_variable_index = variable_diff.loc[(variable_diff>pd.Timedelta('00:30:00')) & (variable_diff<pd.Timedelta('120 days'))].value_counts().sort_index().index
                    gap_cumulative = gaps_variable_index/pd.Timedelta('00:30:00')-1
                    gaps_variable_index = [str(i) for i in gaps_variable_index]
                    gaps_variable_count = variable_diff.loc[(variable_diff>pd.Timedelta('00:30:00')) & (variable_diff<pd.Timedelta('120 days'))].value_counts().sort_index().values

                ax2 = ax[j].twinx()

                gaps_sizes = ax[j].bar(gaps_variable_index, gaps_variable_count)
                # plt.xticks(rotation=90)
                ax[j].set_xticklabels(labels=gaps_variable_index,rotation=90)
                ax[j].set_title(f'{variable} GAPS')
                for i, valor in enumerate(gaps_variable_count):
                    ax[j].text(i-0.5, valor+1, str(valor))

                ax2.plot(gaps_variable_index, gap_cumulative*gaps_variable_count, color='red',linestyle='--')

            fig_01.tight_layout()

        if 'gaps_iab1' in stats:
            import matplotlib.dates as md
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"

            fig_02, ax2 = plt.subplots(1, figsize=(10,4), dpi=300)

            # print(self.iab1_df[['TIMESTAMP','RH']])
            iab1_data = self.iab1_df.loc[self.iab1_df['TIMESTAMP']>'2018-10-05',['TIMESTAMP', 'RH']].copy()
            c = iab1_data.set_index('TIMESTAMP')
            c.resample('D').count()['RH'].plot(ax=ax2, label='Umidade relativa')
            ax2.set_ylim((-1,150))
            # ax2.legend(['dfasdf'])
            ax2.set_ylabel('Quantidade de dados diário')
            ax2.set_xlabel('')
            # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.legend()
            plt.rc('axes', labelsize=14)
            # fig_02.savefig('iab1_gaps_meteo.png')

        if 'gaps_iab2' in stats:
            import matplotlib.dates as md
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"

            fig_03, ax3 = plt.subplots(1, figsize=(10,4), dpi=300)
            columns_iab2 = ['AirTC_Avg','CNR_Wm2_Avg','G_Wm2_Avg']
            iab2_data = self.iab2_df.loc[self.iab2_df['TIMESTAMP']>'2018-10-05', columns_iab2+['TIMESTAMP']].copy()
            # print(self.iab2_df)
            # print(iab2_data)
            d = iab2_data.set_index('TIMESTAMP')
            d.resample('D').count()[columns_iab2[0]].plot(ax=ax3, linestyle='solid',label='Temperatura do ar')
            d.resample('D').count()[columns_iab2[1]].plot(ax=ax3,linestyle='dashed',label='Radiação líquida')
            d.resample('D').count()[columns_iab2[2]].plot(ax=ax3, linestyle='dotted',label='Fluxo de calor do solo')


            ax3.set_ylim((-1,150))
            ax3.set_ylabel('Quantidade de dados diário')
            ax3.set_xlabel('')

            ax3.legend()
            plt.rc('axes', labelsize=14)

            fig_03.savefig('iab2_gaps_meteo.png')

        if 'pm' in stats:
            # plt.rcParams.update({'font.size': 12})

            # print(self.iab3_df[['TIMESTAMP','ga','gc']])
            # print(self.iab3_df['TIMESTAMP'].dt.hour.unique())

            # for hour in self.iab3_df['TIMESTAMP'].dt.hour.unique():
            #     print(self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.hour==hour, ['TIMESTAMP','ga','gc']].mean())
            # print(self.iab3_df.groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['ga','gc'].mean())
            #
            # print(self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==2019)&
            #                  (self.iab3_df['flag_qaqc']==1)&
            #                  (self.iab3_df['flag_rain']==1)&
            #                  (self.iab3_df['flag_signalStr']==1)&
            #                  (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['TIMESTAMP','gc'].var())
            ano = 2019
            meses_chuva = [1,2,3,10,11,12]
            meses_seco = [4,5,6,7,8,9]

            # footprint = 0.8
            fig_01, ax = plt.subplots(2, figsize=(8,6))

            # self.iab3_df.loc[self.iab3_df['ga']>0].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['ga'].mean().plot(ax=ax[0])
            self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                             # (self.iab3_df['flag_qaqc']==1)&
                             # (self.iab3_df['flag_rain']==1)&
                             # (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==1)&
                             (self.iab3_df['ga']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['ga'].mean().plot(ax=ax[0])
            self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                             # (self.iab3_df['flag_qaqc']==1)&
                             # (self.iab3_df['flag_rain']==1)&
                             # (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==0)&
                             (self.iab3_df['ga']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['ga'].mean().plot(ax=ax[0])
            ax[0].set_yscale('log')
            ax[0].set_title(f'{ano} - ga mean per hour')
            ax[0].set_xlim((0,23))


            self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                             (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                             (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                             (self.iab3_df['flag_qaqc']==1)&
                             (self.iab3_df['flag_rain']==1)&
                             (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==1)&
                             (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().plot(ax=ax[1], color='blue', label='$Cerrado\ sensu\ stricto$')
            self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                              (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                              (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                             (self.iab3_df['flag_qaqc']==1)&
                             (self.iab3_df['flag_rain']==1)&
                             (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==0)&
                             (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().plot(ax=ax[1], color='orange', label='$Outros$')

            gc_seco = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                                        (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                                        (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                                         (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_seco))&
                                         (self.iab3_df['flag_qaqc']==1)&
                                         (self.iab3_df['flag_rain']==1)&
                                         (self.iab3_df['flag_signalStr']==1)&
                                         (self.iab3_df['flag_footprint']==1)&
                                         (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().values
            gc_chuva = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                             (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_chuva))&
                             (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                             (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                             (self.iab3_df['flag_qaqc']==1)&
                             (self.iab3_df['flag_rain']==1)&
                             (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==1)&
                             (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().values
            # ax[1].plot(range(6,19,1), gc_seco, linestyle='--', color='lightblue')
            # ax[1].plot(range(6,19,1), gc_chuva, linestyle='--', color='darkblue')

            ax[1].fill_between(range(6,19,1), gc_seco, gc_chuva, alpha=0.2, color='blue')

            gc_seco_p = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                                         (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_seco))&
                                          (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                                          (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                                         (self.iab3_df['flag_qaqc']==1)&
                                         (self.iab3_df['flag_rain']==1)&
                                         (self.iab3_df['flag_signalStr']==1)&
                                         (self.iab3_df['flag_footprint']==0)&
                                         (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().values
            gc_chuva_p = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
                                           (self.iab3_df['TIMESTAMP'].dt.hour>=6)&
                                           (self.iab3_df['TIMESTAMP'].dt.hour<=18)&
                             (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_chuva))&
                             (self.iab3_df['flag_qaqc']==1)&
                             (self.iab3_df['flag_rain']==1)&
                             (self.iab3_df['flag_signalStr']==1)&
                             (self.iab3_df['flag_footprint']==0)&
                             (self.iab3_df['gc']>0)].groupby(by=self.iab3_df['TIMESTAMP'].dt.hour)['gc'].mean().values

            # ax[1].plot(range(6,19,1), gc_seco_p, linestyle='--', color='yellow')
            # ax[1].plot(range(6,19,1), gc_chuva_p, linestyle='--', color='darkorange')

            ax[1].fill_between(range(6,19,1), gc_seco_p, gc_chuva_p, alpha=0.2, color='orange')

            ax[1].set_yscale('log')
            # ax[1].set_title(f'{ano} - gc mean per hour ')
            ax[1].set_ylim((0.001,0.05))

            ax[1].set_ylabel('$g_c [m/s]$')
            ax[1].set_xlabel('Horário do dia')
            ax[1].set_xlim((6,18))
            ax[1].legend()




            # sns.boxplot(x=self.iab3_df['TIMESTAMP'].dt.hour,y='gc',
            #             data=self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
            #                                   (self.iab3_df['flag_qaqc']==1)&
            #                                   (self.iab3_df['flag_rain']==1)&
            #                                   (self.iab3_df['flag_signalStr']==1)&
            #                                   (self.iab3_df['gc']>0)], hue='flag_footprint',ax=ax[2],hue_order=[1,0])
            # ax[2].set_yscale('log')
            # ax[2].set_title(f'{ano} - gc boxplot per hour')
            # ax[2].set_ylim((0.00001,0.1))

            # print(self.iab3_df['TIMESTAMP'].dt.month.isin([1,2,3,8]))

            # sns.boxplot(x=self.iab3_df['TIMESTAMP'].dt.hour, y='gc',
            #             data=self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
            #                                   (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_chuva))&
            #                                   (self.iab3_df['flag_qaqc']==1)&
            #                                   (self.iab3_df['flag_rain']==1)&
            #                                   (self.iab3_df['flag_signalStr']==1)&
            #                                   (self.iab3_df['gc']>0)], hue='flag_footprint',ax=ax[3],hue_order=[1,0])
            # sns.boxplot(x=self.iab3_df['TIMESTAMP'].dt.hour, y='gc',
            #             data=self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==ano)&
            #                                   (self.iab3_df['TIMESTAMP'].dt.month.isin(meses_seco))&
            #                                   (self.iab3_df['flag_qaqc']==1)&
            #                                   (self.iab3_df['flag_rain']==1)&
            #                                   (self.iab3_df['flag_signalStr']==1)&
            #                                   (self.iab3_df['gc']>0)], hue='flag_footprint',ax=ax[4],hue_order=[1,0])

            # ax[3].set_yscale('log')
            # ax[4].set_yscale('log')
            # ax[3].set_ylim((0.00001,0.1))
            # ax[4].set_ylim((0.00001,0.1))
            # ax[3].set_title(f'{ano} - gc boxplot {meses_chuva}')
            # ax[4].set_title(f'{ano} - gc boxplot {meses_seco}')


            # fig_01, ax = plt.subplots(1, figsize=(10,3))
            #
            # b = iab3_df.set_index('TIMESTAMP')
            # b.resample('D').count()[['ga','gc']].plot(ax=ax)
            # ax.set_title('Gaps in ga and gc (IAB3)')
            #
            # self.fitting_gagc(show_graphs=True)
            #
            fig_01.tight_layout()
            fig_01.savefig(f'gagc_{ano}.png', dpi=300)

    def stats_ET(self, stats=[]):
        if 'sum' in stats:
            fig_01, ax = plt.subplots(3, figsize=(10,8))
            # print(self.filled_ET)
            # print(self.iab3_ET_timestamp[self.filled_ET].dtypes)
            # print(self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[['ET_pm_and_ET']+['ET']].sum())
            # print(self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)['ET_pm_and_ET'].sum())
            # print()
            self.iab3_ET_timestamp.sort_values(by='TIMESTAMP', inplace=True)
            try:
                self.iab3_ET_timestamp.reset_index(inplace=True)
            except:
                pass
            # print(self.iab3_ET_timestamp)
            (self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2).plot.bar(ax=ax[0])
            ax[0].set_title('ET yearly sum')
            ax[0].grid(zorder=0)

            (self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].cumsum()/2).set_index(self.iab3_ET_timestamp['TIMESTAMP']).plot(ax=ax[1])
            ax[1].set_title('Cumulative ET yearly sum in a timeseries')

            (self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].count()/17520).plot.bar(ax=ax[2])
            ax[2].set_title('Percentage data filled')

            fig_01.tight_layout()

        if 'gaps' in stats:
            without_bigGAP = True
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"
            fig_01, ax = plt.subplots(len(self.filled_ET)+2, figsize=(10,100), dpi=300)
            b = self.iab3_ET_timestamp.set_index('TIMESTAMP')
            b.resample('D').count()[self.filled_ET+['ET']].plot(ax=ax[0], figsize=(10,5*len(self.filled_ET)+4))
            ax[0].set_ylim((0,50))
            ax[0].set_title('Count of gaps')

            for j, variable in enumerate(['ET']+self.filled_ET):
                # print(j, variable)

                sorted_timestamp = self.iab3_ET_timestamp.sort_values(by='TIMESTAMP')
                et_diff = sorted_timestamp.loc[(sorted_timestamp[f'{variable}'].notna()), 'TIMESTAMP'].diff()

                # print(sorted_timestamp.loc[sorted_timestamp[f'{variable}'].notna(), ['TIMESTAMP',variable]].describe())

                if without_bigGAP == False:
                    gaps_et_index = et_diff.loc[et_diff>pd.Timedelta('00:30:00')].value_counts().sort_index().index
                    gap_cumulative = gaps_et_index/pd.Timedelta('00:30:00')-1
                    gaps_et_index = [str(i-pd.Timedelta('00:30:00')) for i in gaps_et_index]
                    gaps_et_count = et_diff.loc[et_diff>pd.Timedelta('00:30:00')].value_counts().sort_index().values

                elif without_bigGAP == True:
                    gaps_et_index = et_diff.loc[(et_diff>pd.Timedelta('00:30:00')) & (et_diff<pd.Timedelta('120 days'))].value_counts().sort_index().index
                    gap_cumulative = gaps_et_index/pd.Timedelta('00:30:00')-1
                    gaps_et_index = [str(i-pd.Timedelta('00:30:00')) for i in gaps_et_index]
                    gaps_et_count = et_diff.loc[(et_diff>pd.Timedelta('00:30:00')) & (et_diff<pd.Timedelta('120 days'))].value_counts().sort_index().values
                    # print(et_diff.loc[(et_diff>pd.Timedelta('00:30:00')) & (et_diff<pd.Timedelta('120 days'))].value_counts())
                    # print(gaps_et_count)

                ax2 = ax[j+1].twinx()

                gaps_sizes = ax[j+1].bar(gaps_et_index, gaps_et_count)
                # plt.xticks(rotation=90)
                ax[j+1].set_xticklabels(labels=gaps_et_index,rotation=90, fontsize=10)
                ax[j+1].set_title(f'{variable} GAPS')
                ax[j+1].set_ylabel('Número de falhas')
                ax2.set_ylabel('Porcentagem acumulada das falhas')
                ax[j+1].set_xlabel('Tamanho da janela de falha')
                plt.grid()
                for i, valor in enumerate(gaps_et_count):
                    ax[j+1].text(i-0.2, valor+1, str(valor))

                # print(type(gap_cumulative))
                print((gap_cumulative.to_numpy()*gaps_et_count*0.5).cumsum())
                ax2.plot(gaps_et_index, (gap_cumulative.to_numpy()*gaps_et_count*0.5).cumsum()*100/(gap_cumulative.to_numpy()*gaps_et_count*0.5).cumsum()[-1], color='red',linestyle='--')
                # plt.legend()

                # ax[j+1].set_xticklabels([])

            fig_01.tight_layout()
            # fig_01.savefig('gaps_et_methods.png')

        if 'hourly' in stats:
            # print(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour.unique())

            # print(self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp['TIMESTAMP'].dt.year==2019].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour)[self.filled_ET+['ET']].count())
            # self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp['TIMESTAMP'].dt.year==2019].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour)[self.filled_ET+['ET']].count().plot()
            fig_01, ax = plt.subplots(len(self.iab3_ET_timestamp['TIMESTAMP'].dt.year.unique()), figsize=(10,3*len(self.iab3_ET_timestamp['TIMESTAMP'].dt.year.unique())))

            for i, year in enumerate(self.iab3_ET_timestamp['TIMESTAMP'].dt.year.unique()):
                ax[i].set_title(f'Ano {year} - Quantidade de dados por hora no ano')
                self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp['TIMESTAMP'].dt.year==year].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour)[self.filled_ET+['ET']].count().plot(ax=ax[i])

                if calendar.isleap(year):
                    ax[i].set_ylim((0, 732))
                else:
                    ax[i].set_ylim((0, 730))

            fig_01.tight_layout()

        if 'corr' in stats:
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"
            # print(self.iab3_ET_timestamp[['ET_rfr','ET_lr']].describe())
            r = 0
            if 'ET_baseline' in self.ET_names:
                r = 1
                # print('base')
                self.ET_names.remove('ET_baseline')
                # print(self.ET_names)

            # print(self.ET_names+['ET_baseline'])

            corr = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna()), ['ET']+self.ET_names].corr()
            print(corr)

            # print(self.iab3_ET_timestamp.columns)

            self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=6)&
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<18),'Diurno'] = True
            self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<6)|
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=18),'Diurno'] = False


            # print(self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna()), ['ET']+self.ET_names])

            # Talvez fazer uma coluna para diferenciação da hora do dia/ano
            # sns.pairplot(data=self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna()), ['ET','daytime']+self.ET_names],
            #              plot_kws={'alpha': 0.2},
            #              hue='daytime',
            #              hue_order=[1,0],
            #              palette=['orange','blue'],
            #              # hue='flag_footprint',
            #              corner=True)

            for et_names in self.ET_names:
                # fig, ax = plt.subplots(1, figsize=(6,6))
                g = sns.pairplot(data=self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna()), ['ET','Diurno']+[et_names]],
                             plot_kws={'alpha': 0.2},
                             hue='Diurno',
                             hue_order=[1,0],
                             palette=['orange','blue'],
                             # hue='flag_footprint',
                             corner=True, height=3)
                # g.axes[0,0].set_xlim((0,1.1))
                # g.axes[0,0].set_ylim((0,1.1))
                g.axes[1,0].set(ylim=(0,1))
                g.axes[0,0].set(xlim=(0,1))

                # g.savefig(f'corr_{et_names}_adjusted.png', dpi=300)
                # g.fig.set
                # fig = g.get_figure()
                # sns.jointplot(data=self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna()), ['ET','Diurno']+[et_names]],
                #               x='ET', y=et_names,
                #               alpha=0.2,
                #               hue='Diurno'
                              # hue_order=[1,0]
                              # kind='kde'
                              # )

                # fig.show()


            if r == 1:
                # print('dafsfsf')
                self.ET_names = ['ET_baseline'] + self.ET_names

        if 'corr_baseline' in stats:
            corr = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].isna()), self.ET_names].corr()
            print(corr)
            print(self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].isna()), self.ET_names].describe())

            self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=6)&
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<18),'daytime'] = True
            self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<6)|
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=18),'daytime'] = False

            sns.pairplot(data=self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].isna()), ['daytime']+self.ET_names],
                         plot_kws={'alpha': 0.2},
                         hue='daytime',
                         hue_order=[1,0],
                         palette=['orange','blue'],
                         # hue='flag_footprint',
                         corner=True)

        if 'heatmap' in stats:

            import matplotlib.dates as md
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["font.family"] = "Times New Roman"

            self.iab3_ET_timestamp.sort_values(by='TIMESTAMP', inplace=True)
            try:
                self.iab3_ET_timestamp.reset_index(inplace=True)
            except:
                pass

            print(self.ET_names)
            print(self.iab3_ET_timestamp[self.ET_names].max())

            self.iab3_ET_timestamp['date'] = self.iab3_ET_timestamp['TIMESTAMP'].dt.date
            # print(self.iab3_ET_timestamp['date'])

            self.iab3_ET_timestamp['time'] = self.iab3_ET_timestamp['TIMESTAMP'].dt.time

            dnum = mdates.date2num(self.iab3_ET_timestamp['TIMESTAMP'])
            start = dnum[0] - (dnum[1]-dnum[0])/2.
            stop = dnum[-1] + (dnum[1]-dnum[0])/2.
            extent = [start, stop, -0.5, 2]

            # print(pd.to_datetime(self.iab3_ET_timestamp['date']))
            # print(self.iab3_ET_timestamp[['date','time','ET']])
            # print(self.iab3_ET_timestamp.pivot('time','date','ET'))

            fig, ax = plt.subplots(len(self.ET_names)+1,figsize=(8,4*len(self.ET_names)), dpi=300)
            # fig, ax = plt.subplots(1,figsize=(7,3), dpi=300)
            # sns.heatmap(self.iab3_ET_timestamp.pivot('time','date','ET'),fmt='d',
            #             cmap='inferno',
            #              xticklabels=30,
            #              # xticklabels=['2019-08-01','2019-09-30'],
            #              ax=ax)

            im=ax[0].imshow(self.iab3_ET_timestamp.pivot('time','date','ET'), aspect='auto', extent=extent, cmap='viridis', vmin=0, vmax=0.9)
            ax[0].set_title('ET')
            ax[0].xaxis.set_major_locator(mdates.YearLocator())
            ax[0].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax[0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
            ax[0].set_yticklabels(['22:00','20:00','18:00','16:00','14:00','12:00','10:00','08:00','06:00','04:00','02:00','00:00'])
            ax[0].set_yticks(np.linspace(-0.3, 2,12))
            plt.colorbar(im, ax=ax[0])


            # ax[0].set_xticklabels()
            # ax[0].xaxis.set_major_locator(md.YearLocator())
            # ax[0].xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
            # ax[0].xaxis.set_minor_locator(md.DayLocator(interval = 1))
            # ax[0].xaxis.set_major_locator(md.MonthLocator())
            # ax[0].xaxis.set_minor_locator(md.DayLocator())
            # ax[0].xaxis.set_major_formatter(md.DateFormatter('%b'))
            # ax[0].set_major_locator


            for i, et_name in enumerate(self.ET_names):
                # sns.heatmap(self.iab3_ET_timestamp.pivot('time','date',f'{et_name}'),
                #             fmt='d',
                #             cmap='inferno',
                #             xticklabels=30,
                #             ax=ax[i+1])
                im2 = ax[i+1].imshow(self.iab3_ET_timestamp.pivot('time','date',f'{et_name}'), aspect='auto', extent=extent, vmin=0, vmax=0.9)
                # ax[i+1].set_title('ET')
                ax[i+1].xaxis.set_major_locator(mdates.YearLocator())
                ax[i+1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))
                ax[i+1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                ax[i+1].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
                ax[i+1].set_yticklabels(['22:00','20:00','18:00','16:00','14:00','12:00','10:00','08:00','06:00','04:00','02:00','00:00'])
                ax[i+1].set_yticks(np.linspace(-0.3, 2,12))
                ax[i+1].set_title(f'{et_name}', fontsize=5)
                plt.colorbar(im2, ax=ax[i+1])
            # fig.autofmt_xdate()

            fig.tight_layout()
            # fig.savefig('heatmap_et_testes.png', dpi=300)

        if 'daynight' in stats:
            t = self.iab3_ET_timestamp.copy()


            print(self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2)


            # DIVIDIR POR 2, PQ EM CADA HORA TEM 2 VALORES DE ET (mm/h)

            print(self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=6)&
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<=18)].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2)

            t.loc[t['flag_rain']==0, self.filled_ET] = 0
            # print(t.loc[t['flag_rain']==0])
            print(t.loc[(t['TIMESTAMP'].dt.hour>=6)&(t['TIMESTAMP'].dt.hour<=18)].groupby(t['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2)

            # print(t.loc[t['flag_rain']==0, self.filled_ET].count())
        if 'daytime' in stats:
            t = self.iab3_ET_timestamp.copy()
            print(t[self.filled_ET+['ET']].describe())
            print(t.loc[(t['TIMESTAMP'].dt.hour>=6)&(t['TIMESTAMP'].dt.hour<18),self.filled_ET+['ET']].describe())

        if 'daily' in stats:

            meses_chuva = [1,2,3,10,11,12]
            meses_seco = [4,5,6,7,8,9]

            # print(self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].count()/48)
            n_days = self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].count()/48

            et_sum = self.iab3_ET_timestamp.groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2

            print('ET (mm/day) yearly mean')
            print(et_sum/n_days)

            n_days_chuva = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.month.isin(meses_chuva))].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].count()/48
            n_days_seco = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.month.isin(meses_seco))].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].count()/48
            # print('N DAYS')
            # print(n_days)
            # print(n_days_chuva)
            # print(n_days_seco)

            et_sum_chuva = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.month.isin(meses_chuva))].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2
            et_sum_seco = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.month.isin(meses_seco))].groupby(self.iab3_ET_timestamp['TIMESTAMP'].dt.year)[self.filled_ET+['ET']].sum()/2

            print('##############Rainy:')
            print(et_sum_chuva/n_days_chuva)

            print('###############Dry:')
            print(et_sum_seco/n_days_seco)

            fig, ax = plt.subplots(3, figsize=(10,9))

            (et_sum/n_days).plot.bar(ax=ax[0])
            (et_sum_seco/n_days_seco).plot.bar(ax=ax[1])
            (et_sum_chuva/n_days_chuva).plot.bar(ax=ax[2])

            fig.tight_layout()

        if 'error' in stats:
            print()
            for i in self.ET_names:
                print(i)
                a = self.iab3_ET_timestamp.loc[(self.iab3_ET_timestamp['ET'].notna())&(self.iab3_ET_timestamp[i].notna())]
                print(a[['ET',f'{i}']].describe())
                mae = mean_absolute_error(a['ET'], a[i])
                mbe = self.MBE(a['ET'], a[i])
                rmse = (mean_squared_error(a['ET'], a[i]))**(1/2)

                print(f'MAE\t: {mae}')
                print(f'MBE\t: {mbe}')
                print(f'RMSE\t: {rmse}')

        if 'normality' in stats:
            df = self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[['ET']+self.ET_names].notna().sum(axis=1)==len(['ET']+self.ET_names), ['ET']+self.ET_names]
            df_dia = df.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=6)&
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<18)]

            df_noite = df.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<6)|
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=18)]
            for i in df:
                print(i)
                # print(normaltest(df[i]))
                print('DIA: \t', normaltest(df_dia[i]))
                print('NOITE: \t', normaltest(df_noite[i]))
                # print(shapiro(df[i]))
                # print(kstest(df.loc[:500,i], 'norm'))

        if 'variance' in stats:
            # print(self.iab3_ET_timestamp[['ET']+self.ET_names].notna().sum(axis=1))
            print(self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[['ET']+self.ET_names].notna().sum(axis=1)==len(['ET']+self.ET_names), ['ET']+self.ET_names].describe())

            df = self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[['ET']+self.ET_names].notna().sum(axis=1)==len(['ET']+self.ET_names)]

            f, p = f_oneway(*list(df[f'{i}'].values for i in set(df[['ET']+self.ET_names])))

            print(f)
            print(p)

            print(np.concatenate(df[['ET']+self.ET_names].values))
            print(np.tile(['ET']+self.ET_names, len(df)))
            tukey = pairwise_tukeyhsd(endog=np.concatenate(df[['ET']+self.ET_names].values),
                                      groups=np.tile(['ET']+self.ET_names, len(df)),
                                      alpha=0.05)
            print(tukey)

        if 'variance_nonNormal' in stats:
            df = self.iab3_ET_timestamp.loc[self.iab3_ET_timestamp[['ET']+self.ET_names].notna().sum(axis=1)==len(['ET']+self.ET_names)]

            # df = self.iab3_ET_timestamp.copy()


            df_dia = df.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=6)&
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<18)]

            df_noite = df.loc[(self.iab3_ET_timestamp['TIMESTAMP'].dt.hour<6)|
                                       (self.iab3_ET_timestamp['TIMESTAMP'].dt.hour>=18)]
            # print(df.describe())

            # print(kruskal(*list(df[f'{i}'] for i in set(df[['ET']+self.ET_names]))))

            # print(friedmanchisquare(*list(df[f'{i}'].values for i in set(df[['ET']+self.ET_names]))))
            # print(friedmanchisquare(df['ET'],df['ET_mdv_[3, 5]'],df['ET_baseline']))
            for i in self.ET_names:
                print(i)
                # print('DIA: ')
                # print('DIA: \t', kruskalwallis(df_dia['ET'].values, df_dia[i].values))
                # print('NOITE: \t', kruskalwallis(df_noite['ET'].values, df_noite[i].values))
                # print(friedmanchisquare(df['ET'], df[i]))

                print('DIA:\t', kruskal(df_dia['ET'].values, df_dia[i].values))
                print('NOITE: \t', kruskal(df_noite['ET'].values, df_noite[i].values))

                # print(kruskal(df_dia.loc[df_dia['ET'].notna(),'ET'].values, df_dia.loc[df_dia[i].notna(),i].values))

                # print(kruskal(df_dia.loc[(df_dia['ET'].notna())&(df_dia[f'{i}'].notna()), 'ET'].values,
                #               df_dia.loc[(df_dia['ET'].notna())&(df_dia[f'{i}'].notna()), f'{i}'].values))


if __name__ == '__main__':
    gapfilling_iab3(ep_path=r'G:\Meu Drive\USP-SHS\Resultados_processados\EddyPro_Fase010203',
                    lf_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB3',
                    iab1_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB1\IAB1',
                    iab2_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB2\IAB2')
