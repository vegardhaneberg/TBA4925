from datetime import datetime
from calendar import monthrange, month_name
from pathlib import Path

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

import time
import pickle
import math

# Interpolation
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import scipy.interpolate.interpnd

from tqdm import tqdm
tqdm.pandas()

def compute_prn_to_block_value(prn_code):
    iir_list = [2, 13, 16, 19, 20, 21, 22, 28]
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


def generate_qf_list(qf_number):
    qf_list = []
    binary = format(qf_number, 'b')
    for i in range(len(binary)):
        if binary[i] == '1':
            qf_list.append(2 ** (int(i)))

    return qf_list


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


def filter_nan_smap_sm(df):
    try:
        df['smap_sm'] = df['smap_sm'].apply(lambda x: x.item(0))
    except:
        print('SMAP_SM value was already of type: float')
    df = df.dropna()
    return df


def filter_nan_smap_vo(df):
    try:
        df['smap_vo'] = df['smap_vo'].apply(lambda x: x.item(0))
    except:
        print('SMAP_VO value was already of type: float')
    df = df.dropna()
    return df


def filter_nan_smap_sr(df):
    try:
        df['smap_surface_roughness'] = df['smap_surface_roughness'].apply(lambda x: x.item(0))
    except:
        print('SMAP_Surface_Roughness value was already of type: float')
    df = df.dropna()
    return df