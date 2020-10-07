import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


class gapfilling_iab3:
    def __init__(self, ep_path, lf_path, iab2_path, iab1_path):
        # File's Path
        self.iab3EP_path = pathlib.Path(ep_path)
        self.iab3LF_path = pathlib.Path(lf_path)
        self.iab2_path = pathlib.Path(iab2_path)
        self.iab1_path = pathlib.Path(iab1_path)

        self._read_files()

        # self._gagc()

    def _read_files(self):
        # Reading csv files
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

        # Removing duplicated files based on 'TIMESTAMP'
        for df in iab_dfs:
            print('Duplicatas: ',df.duplicated().sum())
            df.drop_duplicates(subset='TIMESTAMP', keep='first', inplace=True)
            df.reset_index(inplace=True)
            print('Verificacao de Duplicatas: ', df.duplicated().sum())

        # Merging files from EddyPro data and LowFreq data
        self.iab3_df = pd.merge(left=self.iab3EP_df, right=self.iab3LF_df, on='TIMESTAMP', how='inner')

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

    def dropping_bad_data(self):
        # Apply filters
        self._applying_filters()

        # Creating a copy and changing to 'nan' filtered values
        iab3_df_copy = self.iab3_df.copy()
        iab3_df_copy.loc[
            (iab3_df_copy['flag_qaqc']==0)|
            (iab3_df_copy['flag_rain']==0)|
            (iab3_df_copy['flag_signalStr']==0)|
            (iab3_df_copy['LE']<0), 'ET'] = np.nan

        return iab3_df_copy

    def _adjacent_days(self, df,n_days=5):
        # Selecting datetime adjectent
        delta_days = [i for i in range(-n_days, n_days+1, 1)]
        df[f'timestamp_adj_{n_days}'] = df['TIMESTAMP'].apply(lambda x: [x + dt.timedelta(days=i) for i in delta_days])

    def _gagc(self):
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

    def dnn_model(self, train_X, val_X, train_y, val_y, learning_rate, epochs, batch_size):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

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
                                  metrics=['mae'])
                    history = model.fit(x=train_X, y=train_y,
                                        epochs=e, batch_size=b, verbose=0,
                                        validation_data=(val_X, val_y))
                    last_mae_t = history.history['mae'][-1]
                    last_mae_v = history.history['val_mae'][-1]

                    plt.title(f'Batch_size: {b} | LR: {l:.2f} | MAE_t: {last_mae_t:.4f} | MAE_v: {last_mae_v:.4f}')
                    plt.plot(history.history['mae'], label='Training')
                    plt.plot(history.history['val_mae'], label='Validation')

                    plt.legend(loc='best')
                    plt.xlabel('# Epochs')
                    plt.ylabel('MAE')
                    # plt.savefig(r'G:\Meu Drive\USP-SHS\Resultados_processados\Gapfilling\ANN\imgs\dnn\{}-epochs_{}-lr_{}-bs.png'.format(e,l,b))
                    plt.show()
                    tf.keras.backend.clear_session()
                    tf.random.set_seed(51)

    def lstm_univariate_model(self, length, generator_train, generator_val, epochs=10):
        tf.keras.backend.clear_session():
        tf.keras.set_seed(51)

        model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=0, input_shape=(length, 1)),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.Hubber(), optimizer='adam', metrics=['mae'])
        history = model.fit(generator_train, epochs=epochs, validation_data=generator_train)

        plt.title('')
        plt.plot(history.history['mae'], label='Training')
        plt.plot(history.history['val_mae'], label='Validation')

        plt.legend(loc='best')
        plt.xlabel('# Epochs')
        plt.ylabel('MAE')
        plt.show()

        tf.keras.backend.clear_session()
        tf.random.set_seed(51)

    def mdv_test(self, n_days=5):
        # Dropping bad data
        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=['ET'], inplace=True)

        # Splitting Dataframe into two parts
        a, b = train_test_split(iab3_df_copy[['TIMESTAMP','ET']])

        # Creating Dataframe with full TIMESTAMP, based on the Dataframe of good data
        date_range = pd.date_range(start=iab3_df_copy['TIMESTAMP'].min(),
                                   end=iab3_df_copy['TIMESTAMP'].max(),
                                   freq='30min')
        df_date_range = pd.DataFrame({'TIMESTAMP':date_range})

        # Merge and creating DataFrame with first part of good data and full TIMESTAMP
        iab3_alldates = pd.merge(left=df_date_range, right=a, on='TIMESTAMP', how='outer')
        # iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')

        # Changing column name of second part of good data
        # To later be compared with gapfilled data
        b.rename(columns={"ET":'ET_val_mdv'}, inplace=True)

        # Merge DataFrame with second part of good data and full TIMESTAMP
        iab3_alldates = pd.merge(left=iab3_alldates, right=b, on='TIMESTAMP', how='outer')

        # Create new column for datetime adjecent days
        self._adjacent_days(df=iab3_alldates, n_days=n_days)

        # Iterating the 'nan' ET and using the non 'nan' ET and adjecent days for filling the data
        for i, row in iab3_alldates.loc[iab3_alldates['ET'].isna()].iterrows():
            iab3_alldates.loc[i, f'ET_mdv_{n_days}'] = iab3_alldates.loc[(iab3_alldates['TIMESTAMP'].isin(row[f'timestamp_adj_{n_days}']))&
                                                                         (iab3_alldates['ET'].notna()), 'ET'].mean()


        # Creating Dataframe for calculate metrics and removing non filled data
        iab3_metrics = iab3_alldates[['ET_val_mdv',f'ET_mdv_{n_days}']].copy()
        iab3_metrics.dropna(inplace=True)

        # Checking metrics
        print(mean_absolute_error(iab3_metrics['ET_val_mdv'], iab3_metrics[f'ET_mdv_{n_days}']))
        print(iab3_metrics.corr())

    def rfr_test(self):
        column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
        column_x_ET = column_x + ['ET']

        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=column_x_ET, inplace=True)

        X = iab3_df_copy[column_x]
        y = iab3_df_copy['ET']

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        print(val_y)

        et_model_RFR = RandomForestRegressor(random_state=1, criterion='mae', max_depth=2)
        et_model_RFR.fit(train_X, train_y)

        val_prediction_RFR = et_model_RFR.predict(val_X)
        # val_y['ET_rfr'] = val_prediction_RFR

        iab3_metrics = pd.DataFrame({'ET':val_y.values, 'ET_rfr':val_prediction_RFR})

        mae_RFR = mean_absolute_error(val_y, val_prediction_RFR)
        print(mae_RFR)

        print(iab3_metrics.corr())
        # print(val_X.)

    def lr_test(self):
        column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
        column_x_ET = column_x + ['ET']

        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=column_x_ET, inplace=True)

        X = iab3_df_copy[column_x]
        y = iab3_df_copy['ET']

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, shuffle=True)

        lm = LinearRegression()
        model = lm.fit(train_X, train_y)

        print(f'''ET = {column_x[0]}*{model.coef_[0]}
              +{column_x[1]}*{model.coef_[1]}
              +{column_x[2]}*{model.coef_[2]}
              +{column_x[3]}*{model.coef_[3]}
              +{column_x[4]}*{model.coef_[4]}
              +{column_x[5]}*{model.coef_[5]}
              +{column_x[6]}*{model.coef_[6]}
              +{column_x[7]}*{model.coef_[7]}
              +{column_x[8]}*{model.coef_[8]}+{model.intercept_}''')

        lm_prediction = model.predict(val_X)

        mae_lm = mean_absolute_error(val_y, lm_prediction)
        print(mae_lm)

        iab3_metrics = pd.DataFrame({'ET':val_y.values, 'ET_lr':lm_prediction})
        print(iab3_metrics.corr())

    def pm_test(self):
        self._adjusting_input_pm()
        self._gagc()

        pm_inputs_iab3 = ['delta', 'Rn_Avg_MJmh', 'shf_Avg_MJmh', 'air_density', 'VPD_kPa', 'ga','LE_MJmh','psychrometric_kPa', 'gc', 'TIMESTAMP']
        pm_inputs_iab3_ET = pm_inputs_iab3 + ['ET']

        iab3_df_copy = self.dropping_bad_data()

        iab3_df_copy.dropna(subset=pm_inputs_iab3_ET, inplace=True)

        X = iab3_df_copy[pm_inputs_iab3]
        y = iab3_df_copy['ET']

        # print(X)

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, shuffle=True)
        # print(val_X)
        val_X = val_X.copy()
        for i, row in self.iab3_df_gagc.iterrows():
            val_X.loc[(val_X['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(val_X['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'ga_mes'] = row['ga']
            val_X.loc[(val_X['TIMESTAMP'].dt.month==row['TIMESTAMP'].month)&(val_X['TIMESTAMP'].dt.year==row['TIMESTAMP'].year), 'gc_mes'] = row['gc']

        val_X['ET_est_pm'] = (val_X['delta']*(val_X['Rn_Avg_MJmh']-val_X['shf_Avg_MJmh'])+3600*val_X['air_density']*1.013*10**(-3)*val_X['VPD_kPa']*val_X['ga_mes'])/(2.45*(val_X['delta']+val_X['psychrometric_kPa']*(1+val_X['ga_mes']/val_X['gc_mes'])))

        print(mean_absolute_error(val_y, val_X['ET_est_pm']))

        # print(val_X['ET_est_pm'])
        # print(val_y)

        # for i,row in self.iab3_df_gagc.iterrows():
        #     val_X.loc[val_X['TIMESTAMP'].dt.month==row['TIMESTAMP'].month, 'ET_est_pm'] = (val_X['delta']*(val_X['Rn_Avg_MJmh']-val_X['shf_Avg_MJmh'])+3600*val_X['air_density']*1.013*10**(-3)*val_X['VPD_kPa']*row['ga'])/(2.45*(val_X['delta']+val_X['psychrometric_kPa']*(1+row['ga']/row['gc'])))


            # print(val_X.loc[val_X['TIMESTAMP'].dt.month==row['TIMESTAMP'].month])

    def dnn_test(self):
        # Dropping bad data
        iab3_df_copy = self.dropping_bad_data()
        iab3_df_copy.dropna(subset=['ET'], inplace=True)
        column_x = ['Rn_Avg', 'RH', 'VPD','air_temperature', 'air_pressure','shf_Avg(1)','shf_Avg(2)','e','wind_speed']
        column_x_ET = column_x + ['ET']

        X = iab3_df_copy[column_x]
        y = iab3_df_copy['ET']

        X_scale = preprocessing.scale(X)
        train_X, val_X, train_y, val_y = train_test_split(X_scale, y, random_state=1, shuffle=True)

        self.dnn_model(train_X=train_X,
                       val_X=val_X,
                       train_y=train_y,
                       val_y=val_y,
                       learning_rate=[0.5e-1, 1e-2],
                       epochs=[100],
                       batch_size=[512])

    def lstm_univariate_test(self):
        pass




if __name__ == '__main__':
    gapfilling_iab3(ep_path=r'G:\Meu Drive\USP-SHS\Resultados_processados\EddyPro_Fase010203',
                    lf_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB3',
                    iab1_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB1\IAB1',
                    iab2_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB2\IAB2')
