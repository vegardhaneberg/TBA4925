import datetime
import pandas as pd
import netCDF4 as nc
from progressbar import progressbar
import os
import time


def conv(t):
    try:
        return pd.Timestamp(t)
    except:
        return pd.Timestamp(t.split('.')[0] + '.000Z')


def convert_time(df: pd.DataFrame, reference_time=None) -> pd.DataFrame:
    if reference_time is None:
        ref_date = pd.Timestamp('2019-01-01T00:00:00.000Z')
    else:
        ref_date = pd.Timestamp(reference_time)

    df['time'] = df['time'].apply(lambda t: conv(t))
    df['time'] = df['time'].apply(lambda t: (t - ref_date).days * 24 + (t - ref_date).seconds / 3600)
    return df


def get_smap(path: str):
    start = datetime.datetime(2019, 1, 1)

    year = int(path.split('_')[4][:4])
    month = int(path.split('_')[4][4:6])
    day = int(path.split('_')[4][6:])

    current_date = datetime.datetime(year, month, day)

    ds = nc.Dataset(path)
    sm = ds['Soil_Moisture_Retrieval_Data_AM']
    latitudes = []
    longitudes = []
    moistures = []
    times = []
    qfs = []

    for lat in range(len(sm['latitude'])):
        for long in range(len(sm['longitude'][lat])):
            latitudes.append(sm['latitude'][lat][long])
            longitudes.append(sm['longitude'][lat][long])
            moistures.append(sm['soil_moisture'][lat][long])
            times.append(sm['tb_time_utc'][lat][long])
            qfs.append(sm['retrieval_qual_flag'][lat][long])

    df = pd.DataFrame.from_dict(
        {'lat': latitudes, 'long': longitudes, 'time': times, 'smap_sm': moistures, 'retrieval_qfs': qfs})

    # Filter out missing values
    smap_df = df[df['smap_sm'] != -9999.0]

    # Convert time
    smap_df = convert_time(smap_df)

    # Filter on time
    smap_df = smap_df[smap_df['time'] >= (current_date - start).days * 24]
    smap_df = smap_df[smap_df['time'] <= (current_date - start).days * 24 + 24]

    # Grid box
    smap_df = grid_box(smap_df, 0.5, 'smap_sm', use_median=False)

    save_path = 'SMAP ' + str(year) + '/' + str(month) + '-' + str(day) + '.csv'
    smap_df[['lat', 'long', 'smap_sm']].to_csv(save_path, index=False)


def get_smap_main(root_path: str) -> pd.DataFrame:
    first = True
    subdirs = []
    filenames = []

    for dir_name, subdir_list, file_list in os.walk(root_path):
        if first:
            subdirs = subdir_list
            first = False
        else:
            filenames.append(file_list[0])

    for i in progressbar(range(len(subdirs))):

        current_path = root_path + '/' + subdirs[i] + '/' + filenames[i]
        get_smap(current_path)


def round_nearest(x, a):
    return round(x / a) * a


def grid_box(df, resolution, target_value='sr', use_median=False):
    df['lat'] = df['lat'].apply(lambda x: round_nearest(x, resolution))
    df['long'] = df['long'].apply(lambda x: round_nearest(x, resolution))

    if use_median:
        df = df.groupby(['long', 'lat'], as_index=False)[target_value].median()
    else:
        df = df.groupby(['long', 'lat'], as_index=False)[target_value].mean()

    return df


root_path = '/Users/mads/Desktop/Global SMAP Data/'

sub_folders = ['5000003240316-1', '5000003240316-2', '5000003240316-3', '5000003240316-4', '5000003240316-5',
               '5000003240316-6', '5000003240316-7', '5000003240316-8', '5000003240316-9']


for folder in sub_folders:
    print('prosessing subfolder:', folder)
    current_path = root_path + folder + '/'

    get_smap_main(current_path)
