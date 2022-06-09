import numpy as np

def calculate_sr_value(snr, p_r, g_t, g_r, d_ts, d_sr):
    # snr(dB), p_r(dBW), g_t(dBi), g_r(dBi), d_ts(meter), d_sr(meter)
    return snr - p_r - g_t - g_r - (20 * np.log10(0.19)) + (20 * np.log10(d_ts + d_sr)) + (20 * np.log10(4 * np.pi))


def compute_surface_reflectivity(df):
    df['sr'] = df.apply(
        lambda row: calculate_sr_value(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i, row.sp_rx_gain,
                                       row.tx_to_sp_range, row.rx_to_sp_range), axis=1)
    return df


def calculate_hours_after_jan_value(day_of_year, ddm_timestamp):
    return (day_of_year - 1) * 24 + round(ddm_timestamp / (60 * 60))


def compute_hours_after_jan(df):
    df['hours_after_jan_2020'] = df.apply(
        lambda row: calculate_hours_after_jan_value(row.day_of_year, row.ddm_timestamp_utc), axis=1)
    return df


def generate_unique_track_id_value(track_id, day_of_year, prn_nr, sat_nr):
    return track_id * 10000 + prn_nr * 10 + sat_nr + day_of_year/1000


def compute_unique_track_ids(df):
    df['unique_track_id'] = df.apply(
        lambda row: generate_unique_track_id_value(row.track_id, row.day_of_year, row.prn_code, row.spacecraft_num), axis=1)
    return df


def generate_qf_list(qf_number):
    qf_list = []
    binary = format(qf_number, 'b')
    for i in range(len(binary)):
        if binary[i] == '1':
            qf_list.append(2 ** (int(i)))

    return qf_list


def compute_prn_to_block_value(prn_code):
    iir_list = [2, 13, 16, 19, 20, 21, 22]
    iif_list = [1, 3, 6, 8, 9, 10, 25, 26, 27, 30, 32]
    iir_m_list = [5, 7, 12, 15, 17, 29, 31]
    iii_list = [4, 11, 14, 18, 23, 24]
    
    if prn_code in iir_list:
        return 'IIR'
    elif prn_code in iif_list:
        return 'IIF'
    elif prn_code in iir_m_list:
        return 'IIR-M'
    elif prn_code in iii_list:
        return 'III'
    else:
        return 'UNKNOWN'


def compute_block_code(df):
    df['block_code'] = df.apply(lambda row: compute_prn_to_block_value(row.prn_code), axis=1)
    return df


def compute_daily_hour_column(df):
    df['daily_hour'] = df.apply(lambda row: round(row.ddm_timestamp_utc / (60*60)), axis=1)
    return df


def compute_time_of_day_value(time):
    if time >= 22:
        return 'N'
    elif time >= 16:
        return 'A'
    elif time >= 10:
        return 'D'
    elif time >= 4:
        return 'M'
    else:
        return 'N'
    

def compute_time_of_day(df):
    df['time_of_day'] = df.apply(lambda row: compute_time_of_day_value(row.daily_hour), axis=1)
    return df


def scale_sr_values(df):
    min_sr = df['sr'].min()
    df['sr'] = df['sr'].apply(lambda x: x - min_sr)
    return df


def filter_location(df, location):
    filtered_df = df[df.sp_lat < location[0]]
    filtered_df = filtered_df[filtered_df.sp_lat > location[2]]
    filtered_df = filtered_df[filtered_df.sp_lon < location[3]]
    filtered_df = filtered_df[filtered_df.sp_lon > location[1]]
    return filtered_df


def filter_quality_flags_1(df):
    df['qf_ok'] = df.apply(
        lambda row: (2 or 4 or 5 or 8 or 16 or 17) not in generate_qf_list(int(row.quality_flags)), axis=1)
    df = df[df['qf_ok']]
    return df


def filter_quality_flags_2(df):
    res_df = df
    res_df['qf2_ok'] = res_df.apply(
        lambda row: (1 or 2) not in generate_qf_list(int(row.quality_flags_2)), axis=1)  # Remember to check which qfs
    res_df = res_df[res_df['qf2_ok']]
    return res_df


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


# Returning the same df if no specific features are selected
def select_df_features(df, feature_list):
    if len(feature_list) > 0:
        return df[feature_list]
    else:
        return df


def store_df_as_csv(df, storage_path):
    df.to_csv(storage_path, index=False)
