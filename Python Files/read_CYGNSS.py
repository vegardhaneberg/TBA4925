import numpy as np
import pandas as pd


def calculate_sr_value(snr, p_r, g_t, g_r, d_ts, d_sr):
    return snr - p_r - g_t - g_r - (20 * np.log10(0.19)) + (20 * np.log10(d_ts + d_sr)) + (20 * np.log10(4 * np.pi))


def compute_surface_reflectivity(df):
    df['sr'] = df.apply(
        lambda row: calculate_sr_value(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i, row.sp_rx_gain,
                                       row.tx_to_sp_range, row.rx_to_sp_range), axis=1)
    return df


def calculate_hours_after_jan_value(day_of_year, ddm_timestamp):
    return (day_of_year - 1) * 24 + ddm_timestamp / (60 * 60)


def compute_hours_after_jan(df):
    df['hours_after_jan_2020'] = df.apply(
        lambda row: calculate_hours_after_jan_value(row.day_of_year, row.ddm_timestamp_utc), axis=1)
    return df


def filter_cygnss_df(df: pd.DataFrame, area: list) -> pd.DataFrame:
    """
    Filters cygnss dataframe
    :param df: pd.Dataframe
    :param area: [N, W, S, E]
    :return: pd.Dataframe
    """
    new_df = df[df['sp_lat'] < area[0]]
    new_df = new_df[new_df['sp_lat'] > area[2]]
    new_df = new_df[new_df['sp_lon'] > area[1]]
    new_df = new_df[new_df['sp_lon'] < area[3]]

    return new_df


def fuzzy_filter_cygnss_df(df: pd.DataFrame, area: list) -> pd.DataFrame:
    """
    Filters cygnss dataframe for a 10 degrees larger area than the input
    :param df: pd.Dataframe
    :param area: [N, W, S, E]
    :return: pd.Dataframe
    """
    new_df = df[df['sp_lat'] < area[0] + 10]
    new_df = new_df[new_df['sp_lat'] > area[2] - 10]
    new_df = new_df[new_df['sp_lon'] > area[1] - 10]
    new_df = new_df[new_df['sp_lon'] < area[3] + 10]

    return new_df


def get_cygnss_df(cygnss_root_path: str, days: list, area: list, fuzzy_filter=False) -> pd.DataFrame:
    """
    If you want 1st to 3rd of August, days should be [1, 2, 3]
    :param cygnss_root_path: path to the root folder with cygnss data
    :param days: days list, e.g. [1, 2, 3]
    :param area: [N, W, S, E]
    :param fuzzy_filter: boolean. Whether or not to use fuzzy filter.
    :return: pd.Dataframe
    """

    file_start = 'raw_main_df_2020_08_'
    file_ending = 'of31.csv'

    cygnss_df = pd.DataFrame()

    for day in days:
        current_path = cygnss_root_path + file_start + str(day) + file_ending
        if fuzzy_filter:
            current_cygnss_df = compute_surface_reflectivity(fuzzy_filter_cygnss_df(pd.read_csv(current_path), area))
        else:
            current_cygnss_df = compute_surface_reflectivity(filter_cygnss_df(pd.read_csv(current_path), area))
        current_cygnss_df = compute_hours_after_jan(current_cygnss_df)
        cygnss_df = cygnss_df.append(current_cygnss_df)

    return cygnss_df
