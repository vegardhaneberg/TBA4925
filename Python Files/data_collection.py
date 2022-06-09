from pydap.client import open_url
from datetime import datetime
from calendar import monthrange

import time
import numpy as np
import pandas as pd

def generate_url(year, month, day, satellite_number):

    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url

    return data_url + '?sp_lat,sp_lon,ddm_timestamp_utc,ddm_snr,gps_tx_power_db_w,gps_ant_gain_db_i,rx_to_sp_range,' \
                      'tx_to_sp_range,sp_rx_gain,spacecraft_num,prn_code,track_id,quality_flags,quality_flags_2,sp_inc_angle', day_of_year


def collect_dataset(day_of_year, url, satellite_nr):
    dataset = open_url(url, output_grid=False)

    df = pd.DataFrame()
    track_list = []

    for ddm in range(4):  # Remember to change back to 4

        ddm_df = pd.DataFrame()
        print("ddm: " + str(ddm))
        sp_lat = np.array(dataset.sp_lat[:, ddm])
        sp_lon = np.array(dataset.sp_lon[:, ddm])
        a, b = (np.where(sp_lon > 180))
        sp_lon[a] -= 360

        ddm_timestamp_utc = np.array(dataset.ddm_timestamp_utc[:, ddm])
        ddm_snr = np.array(dataset.ddm_snr[:, ddm])
        gps_tx_power_db_w = np.array(dataset.gps_tx_power_db_w[:, ddm])
        gps_ant_gain_db_i = np.array(dataset.gps_ant_gain_db_i[:, ddm])
        rx_to_sp_range = np.array(dataset.rx_to_sp_range[:, ddm])
        tx_to_sp_range = np.array(dataset.tx_to_sp_range[:, ddm])
        sp_rx_gain = np.array(dataset.sp_rx_gain[:, ddm])
        track_id = np.array(dataset.track_id[:, ddm])
        prn_code = np.array(dataset.prn_code[:, ddm])
        quality_flags = np.array(dataset.quality_flags[:, ddm])
        quality_flags_2 = np.array(dataset.quality_flags_2[:, ddm])
        sp_inc_angle = np.array(dataset.sp_inc_angle[:, ddm])

        ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
        ddm_df['spacecraft_num'] = np.zeros(len(sp_lon))
        ddm_df['day_of_year'] = np.zeros(len(sp_lon))
        ddm_df['sp_lat'] = sp_lat.tolist()
        ddm_df['sp_lon'] = sp_lon.tolist()
        ddm_df = ddm_df.assign(ddm_channel=ddm)
        ddm_df = ddm_df.assign(spacecraft_num=satellite_nr)
        ddm_df = ddm_df.assign(day_of_year=day_of_year)

        ddm_df['ddm_timestamp_utc'] = ddm_timestamp_utc.tolist()
        ddm_df['ddm_snr'] = ddm_snr.tolist()
        ddm_df['gps_tx_power_db_w'] = gps_tx_power_db_w.tolist()
        ddm_df['gps_ant_gain_db_i'] = gps_ant_gain_db_i.tolist()
        ddm_df['rx_to_sp_range'] = rx_to_sp_range.tolist()
        ddm_df['tx_to_sp_range'] = tx_to_sp_range.tolist()
        ddm_df['sp_rx_gain'] = sp_rx_gain.tolist()
        ddm_df['track_id'] = track_id.tolist()
        ddm_df['prn_code'] = prn_code.tolist()
        ddm_df['sp_inc_angle'] = sp_inc_angle.tolist()
        ddm_df['quality_flags'] = quality_flags.tolist()
        ddm_df['quality_flags_2'] = quality_flags_2.tolist()

        for col in ddm_df.columns:
            if col != 'ddm_channel' and col != 'ddm_timestamp_utc' and col != 'spacecraft_num' and col != 'day_of_year':
                ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
        df = df.append(ddm_df, ignore_index=True)

    return df


def collect_data(url):
    data = open_url(url, output_grid=False)
    return data


def calculate_sr(snr, p_r, g_t, g_r, d_ts, d_sr):
    # snr(dB), p_r(dBW), g_t(dBi), g_r(dBi), d_ts(meter), d_sr(meter)
    return snr - p_r - g_t - g_r - (20*np.log10(0.19)) + (20*np.log10(d_ts+d_sr)) + (20*np.log10(4*np.pi))


def remove_fill_values(df, raw_data):
    keys = list(raw_data.keys())
    keys.remove('ddm_timestamp_utc')
    keys.remove('spacecraft_num')
    filtered_df = df

    # Remove rows containing fill values
    for k in keys:
        key = raw_data[k]
        fv = key._FillValue
        filtered_df = filtered_df[filtered_df[k] != fv]

    return filtered_df


def filter_bad_quality_flags(df):
    filtered_df = df[(df.quality_flags % 2) == 0]
    return filtered_df


def filter_location(df, location):
    filtered_df = df[df.sp_lat < location[0]]
    filtered_df = filtered_df[filtered_df.sp_lat > location[2]]
    filtered_df = filtered_df[filtered_df.sp_lon < location[3]]
    filtered_df = filtered_df[filtered_df.sp_lon > location[1]]
    return filtered_df


def compute_surface_reflectivity(df):
    df['sr'] = df.apply(lambda row: calculate_sr(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i, row.sp_rx_gain, row.tx_to_sp_range, row.rx_to_sp_range), axis=1)
    return df


def compute_hour(seconds):
    return round(seconds/(60*60))


def compute_nearest_hour(df):
    df['hour'] = df.apply(lambda row: compute_hour(row.ddm_timestamp_utc), axis=1)
    return df


def create_cygnss_df(year, month, day):

    df = pd.DataFrame()
    raw_data_list = []
    failed_satellites = []
    
    sat_counter = 0
    failed_attempts = 0

    while sat_counter < 8:  # Remember to change back to 8 satellites for all data collection
        try:
            satellite_start = time.time()
            print('Starting computations for satellite number ' + str(sat_counter+1) + '...')
            print('------------------------------------------------------------')

            print('Generating url...')
            data_url, day_of_year = generate_url(year, month, day, sat_counter+1)

            print('Collecting data as a DataFrame...')
            satellite_df = collect_dataset(day_of_year, data_url, sat_counter+1)

            print('Collecting raw data...')
            raw_data = collect_data(data_url)
            raw_data_list.append(raw_data)

            seconds = time.time()-satellite_start
            print('Collected data for satellite ' + str(sat_counter+1) + ' in ' + str(round(seconds/60)) + ' minutes and ' + 
                  str(seconds % 60) + ' seconds.')
            print('#####################################################')
            print('#####################################################\n\n')

            df = df.append(satellite_df)
            sat_counter += 1
        except:
            print('Data collection failed. Trying again...')
            failed_attempts += 1
        
        if failed_attempts == 50:
            failed_satellites.append(sat_counter+1)
            sat_counter += 1
            failed_attempts = 0
            print('Data collection aborted. Trying the next satellite!')
            
    return df, raw_data_list, failed_satellites


####COLLECT DATA FOR A NUMBER OF DAYS####
year = 2019
start_month = 11
initial_start_day = 1
first = True

for month in range(start_month, 12): # Change to 13 for the entire year

    num_of_days = monthrange(year, month)[1]
    
    if first:
        start_day = initial_start_day
    else:
        start_day = 1

    if month == 11:
        end_day = 10
    else:
        end_day = num_of_days

    days_with_error = []

    error_month = []
    error_day = []
    error_satellites = []

    raw_main_df = pd.DataFrame()

    for i in range(start_day, end_day+1):  # Number of days to collect data # Change to end_day+1 for entire month
        print('#############################################################')
        print('#############################################################')
        print('Starting computation for day ' + str(i) + ' of '+ str(end_day) + '...........')
        raw_df, raw_data_list, failed_satellites = create_cygnss_df(year, month, i)
        
        if len(failed_satellites) > 0:
            days_with_error.append([i, failed_satellites])
            error_month.append(month)
            error_day.append(i)
            error_satellites.append(failed_satellites)
        
        print('-------------------------------------------------------------')
        
        print('Removing fill values...')
        rows_before_removal = raw_df.shape[0]
        satellite_df = raw_df
        
        for j in range(len(raw_data_list)):  # Number of satellites
            satellite_df = remove_fill_values(raw_df, raw_data_list[j])
        
        rows_after_removal = satellite_df.shape[0]
        print('Removed ' + str(rows_before_removal - rows_after_removal) + ' rows containing fill values...')
        
        satellite_df.to_csv("/Users/mads/Downloads/raw_main_df_" + str(year) + "_" + str(month).zfill(2) + "_" + str(i) + "of" + str(end_day) + ".csv", index=False)
        #satellite_df.to_csv("/Volumes/DACOTA HDD/CYGNSS/raw_main_df_" + str(year) + "_" + str(month).zfill(2) + "_" + str(i) + "of" + str(end_day) + ".csv", index=False)

        print('Day: ', i)
        print('Failed satellites: ', failed_satellites)
        print('Days with error in total: ', days_with_error)
        
        print('#############################################################')
        print('#############################################################\n\n')
    
    error_dict = {'month': error_month, 'day': error_day, 'satellites': error_satellites}
    error_df = pd.DataFrame(error_dict)
    #error_df.to_csv("/Volumes/DACOTA HDD/CYGNSS/FailedSatellites/failed_satellites_" + str(year) + "_" + str(month).zfill(2) + ".csv", index=False)
    error_df.to_csv("/Users/mads/Downloads/failed_satellites_" + str(year) + "_" + str(month).zfill(2) + ".csv", index=False)

    first = False
