import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from sklearn.metrics import mean_squared_error as mse
from time import time

import ReadSMAP
import ReadCYGNSS


def create_interval_list(interval: int, min_value: int, max_value: int) -> list:
    interval_list = []

    while min_value <= max_value:
        interval_list.append(min_value)
        min_value += interval

    return interval_list


def interpolate(df: pd.DataFrame, target_value, lat_name='lat', long_name='long',
                time_name='time') -> LinearNDInterpolator:
    coordinates = list(zip(list(df[time_name]), list(df[lat_name]), list(df[long_name])))
    target = df[target_value]
    interpolation_function = LinearNDInterpolator(coordinates, target)
    return interpolation_function


def parameter_analysis(target_val: str, cygnss_root_path: str, smap_root_path: str, days: list, area: list,
                       interval: int, min_value: int = None, max_value: int = None, classes=None, fuzzy=False):
    # Read CYGNSS
    start = time()
    cygnss_df = ReadCYGNSS.get_cygnss_df(cygnss_root_path, days, area, fuzzy_filter=fuzzy)
    print('Done reading ' + str(len(cygnss_df)) + ' CYGNSS items in ' + str(round(time() - start, 0)) + ' sec')

    # Read SMAP
    start = time()
    smap_df = ReadSMAP.get_smap_df(smap_root_path, 2020, convert_time_hours=True)
    print('Done reading ' + str(len(smap_df)) + ' SMAP items in ' + str(round(time() - start, 0)) + ' sec')

    if classes is None:  # Then we need to create classes
        if min_value is None and max_value is None:  # Then we need to set the boundary values
            if target_val in list(cygnss_df.columns):
                min_value = min(list(cygnss_df[target_val]))
                max_value = max(list(cygnss_df[target_val]))
            else:
                min_value = min(list(smap_df[target_val]))
                max_value = max(list(smap_df[target_val]))
        classes = create_interval_list(interval, min_value, max_value)

    statistics_dict = {}

    print(smap_df.columns)

    for group in classes:

        current_cygnss = cygnss_df[cygnss_df[target_val] >= group]
        if not group == classes[len(classes) - 1]:  # If last bin we dont exclude values larger than the current group
            current_cygnss = current_cygnss[current_cygnss[target_val] < group + interval]
        current_smap = smap_df.copy(deep=True)

        print('Starting on group: ' + str(group) + ' to ' + str(group + interval) + '   Length of cygnss:' +
              str(len(current_cygnss)) + '   Length of SMAP:' + str(len(current_smap)))

        inter_function = interpolate(current_cygnss, 'sr', 'sp_lat', 'sp_lon', 'hours_after_jan_2020')

        current_smap['sr'] = current_cygnss.apply(lambda row: inter_function(row.hours_after_jan_2020, row.sp_lat, row.sp_lon), axis=1)

        corr = current_smap['sr'].corr(current_smap['smap_sm'])
        rmse = np.sqrt(mse(current_smap['sr'], current_smap['smap_sm']))

        statistics_dict[group] = [corr, rmse]

    return statistics_dict


# 1st of August 2020
c_root_path = "/Volumes/DACOTA HDD/Semester Project CSV/CYGNSS 2020-08/"
c_days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
s_path = "/Users/vegardhaneberg/Desktop/Masters Thesis/Code/Master/Data/SMAP/India first two weeks of August"

# area = [N, W, S, E]
test_area = [27.2, 80.32, 21.81, 88.29]

stats = parameter_analysis('sp_inc_angle', c_root_path, s_path, c_days, test_area, 20, 0, 70, fuzzy=False)
print(stats)
