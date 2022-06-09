import pandas as pd
import os
import netCDF4 as nc
from time import time
from progressbar import progressbar


def get_smap(path: str, printing=False):
    ds = nc.Dataset(path)
    sm = ds['Soil_Moisture_Retrieval_Data_AM']

    latitudes = []
    longitudes = []
    moistures = []
    times = []

    for lat in progressbar(range(len(sm['latitude']))):
        for long in range(len(sm['longitude'][lat])):
            latitudes.append(sm['latitude'][lat][long])
            longitudes.append(sm['longitude'][lat][long])
            moistures.append(sm['soil_moisture'][lat][long])
            times.append(sm['tb_time_utc'][lat][long])

    df = pd.DataFrame.from_dict({'lat': latitudes, 'long': longitudes, 'time': times, 'smap_sm': moistures})

    # Filter out missing values
    smap_df = df[df['smap_sm'] != -9999.0]

    if len(smap_df) > 0 and printing:
        print('Number of missing values:', len(df) - len(smap_df))
        print('Number of data points with value:', len(smap_df))
        index = list(smap_df['smap_sm']).index(max(list(smap_df['smap_sm'])))
        print("Peak SM value:", list(smap_df['smap_sm'])[index])
        print("Peak SM value at: (" + str(list(smap_df['lat'])[index]) + ", " + str(list(smap_df['long'])[index]) + ")")

    return smap_df


def conv(t):
    try:
        return pd.Timestamp(t)
    except:
        return pd.Timestamp(t.split('.')[0] + '.000Z')


def convert_time(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = pd.Timestamp('2020-01-01T00:00:00.000Z')

    df['time'] = df['time'].apply(lambda t: conv(t))
    df['time'] = df['time'].apply(lambda t: (t - ref_date).days * 24 + (t - ref_date).seconds / 3600)
    return df


def get_smap_df(root_dir: str, year: int, convert_time_hours=True) -> pd.DataFrame:
    first = True
    all_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if not first:
                all_paths.append(os.path.join(subdir, file))
            else:
                first = False

    smap_df = pd.DataFrame()

    for path in all_paths:
        path_split = path.split('_')
        current_year = int(path_split[4][:4])

        if current_year == year:
            current_df = get_smap(path)
            smap_df = smap_df.append(current_df)

    if convert_time_hours:
        smap_df = convert_time(smap_df)

    return smap_df
