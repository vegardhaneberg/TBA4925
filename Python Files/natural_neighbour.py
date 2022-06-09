import numpy as np
import pandas as pd

import time
import pickle

from tqdm import tqdm
tqdm.pandas()

import naturalneighbor


def filter_cygnss_qf(df):
    remove_mask = int('11000000010011010', 2)
    return df[df['quality_flags'] & remove_mask == 0]


def filter_smap_qfs(smap_df):
    return smap_df.loc[(smap_df['retrieval_qfs'] == 0) | (smap_df['retrieval_qfs'] == 8)]


def get_hours_after_jan_2019(old_hours, year):
    if year == 2019:
        return old_hours
    elif year == 2020:
        return old_hours + 24*365
    else:
        return old_hours + 24*365 + 24*366 # 366 fordi 2020 var skuddår


def fix_hours_after_jan_2019(df):
    df['hours_after_jan_2019'] = df.progress_apply(
        lambda row: get_hours_after_jan_2019(row.hours_after_jan_2020, row.year), axis=1)
    return df


def filter_location(df, location, cygnss=True):
    if cygnss:
        filtered_df = df[df.sp_lat < location[0]]
        filtered_df = filtered_df[filtered_df.sp_lat > location[2]]
        filtered_df = filtered_df[filtered_df.sp_lon < location[3]]
        filtered_df = filtered_df[filtered_df.sp_lon > location[1]]
    else:
        filtered_df = df[df.lat < location[0]]
        filtered_df = filtered_df[filtered_df.lat > location[2]]
        filtered_df = filtered_df[filtered_df.long < location[3]]
        filtered_df = filtered_df[filtered_df.long > location[1]]
    
    return filtered_df



# INDIA USED FOR TESTING
location = '24-80-19-85'
small_pixel_worst = [20, 83.5, 19.5, 84]
small_pixel_best = [21.5, 81.5, 21, 82]

pixel_loc_worst = [20.5, 83, 19, 84.5]
pixel_loc_best = [22, 81, 20.5, 82.5]



### Load CYGNSS ###
cygnss_df = pd.DataFrame()

for year in range(2019, 2022):
    retrieval_path = '/Users/madsrindal/Desktop/Intervals/' + location + '/'
    file_name = 'CYGNSS' + str(year) + '-withQFs-[' + location + '].csv'
    tmp_df = pd.read_csv(retrieval_path + file_name)
    
    # Adding column for year
    tmp_df['year'] = year
    
    shape_before = tmp_df.shape[0]
    tmp_df = filter_cygnss_qf(tmp_df)
    print('Removed ' + str(shape_before - tmp_df.shape[0]) + ' rows of bad CYGNSS data from ' + str(year) + '...')
    cygnss_df = cygnss_df.append(tmp_df, ignore_index=True)

# Fix hours_after_jan_2019 column
cygnss_df = fix_hours_after_jan_2019(cygnss_df)

# Remove unused CYGNSS parameters
cygnss_cols_to_drop = ['ddm_channel', 'ddm_timestamp_utc', 'rx_to_sp_range', 'tx_to_sp_range', 
                       'quality_flags', 'quality_flags_2', 'hours_after_jan_2020']
cygnss_df = cygnss_df.drop(columns=cygnss_cols_to_drop)



### Load SMAP ###
retrieval_path = '/Users/madsrindal/Desktop/Intervals/' + location + '/'
file_name = 'SMAP-allYears-withQFs-[' + location + '].csv'
smap_df = pd.read_csv(retrieval_path + file_name)
shape_before = smap_df.shape[0]
smap_df = filter_smap_qfs(smap_df)
print('Removed ' + str(shape_before - smap_df.shape[0]) + ' rows of bad SMAP data...')

# Remove unused SMAP parameters
smap_cols_to_drop = ['retrieval_qfs', 'surface_temp', 'vegetation_water_content', 'landcover_class_01', 'landcover_class_02', 'landcover_class_03']
smap_df = smap_df.drop(columns=smap_cols_to_drop)

# Filter data based on pixel location
cygnss_pxl_df = filter_location(cygnss_df, pixel_loc_best, True)
smap_pxl_df = filter_location(smap_df, pixel_loc_best, False)



### Natural Neighbour Interpolation ###
def nn_interpolation(smap_pxl_df, grid_ranges, target_value='smap_sm'):
    points = np.array(list(map(list, zip(list(smap_pxl_df['time']), list(smap_pxl_df['lat']), list(smap_pxl_df['long'])))))
    values = np.array(list(smap_pxl_df[target_value]))
    nn_interpolated_values = naturalneighbor.griddata(points, values, grid_ranges)
    return nn_interpolated_values


def create_df_from_3d_array(interpolated_array, grid_ranges, target_value='smap_sm'):
    lat_min = grid_ranges[0][0]
    lat_max = grid_ranges[0][1]
    lat_complex = int(str(grid_ranges[0][2])[:-1])
    lat_step = (lat_max-lat_min) / (lat_complex-1)

    long_min = grid_ranges[1][0]
    long_max = grid_ranges[1][1]
    long_complex = int(str(grid_ranges[1][2])[:-1])
    long_step = (long_max-long_min) / (long_complex-1)

    time_min = grid_ranges[2][0]
    time_max = grid_ranges[2][1]
    time_step = grid_ranges[2][2]
    
    lats = []
    longs = []
    times = []
    values = list(interpolated_array.flatten())

    for i in range(lat_complex):
        lat = lat_min + (i*lat_step)

        for j in range(long_complex):
            long = long_min + (j*long_step)

            for k in range(time_min, time_max, time_step):

                lats.append(lat)
                longs.append(long)
                times.append(k)

    df = pd.DataFrame()
    df['lat'] = lats
    df['long'] = longs
    df['time'] = times
    df[target_value] = values
    return df
    

# Set grid ranges to obtain a SM value each 1km lat and long per day
grid_ranges = [[20.5, 22, 165j], [81, 82.5, 165j], [12, 26316, 24]] # One value per 1km lat and long + each day
# grid_ranges = [[20.5, 22, 16j], [81, 82.5, 16j], [12, 26316, 168]] # One value per 10km lat and long + each 7 days
storage_path = '/Users/madsrindal/Desktop/Natural Neighbour/India/Best Cell/'

### Create interpolated 3D grid with Soil Moisture values ###
start = time.time()
target = 'smap_sm'
nn_interpolated_sm = nn_interpolation(smap_pxl_df, grid_ranges, target_value=target)
print('Finished natural neighbour interpolation with smap_sm as the target value in ' + str(time.time()-start) + ' seconds...')

# Store the raw array as a pickle file
output = open(storage_path + target + '_raw_array.pkl', 'wb')
pickle.dump(nn_interpolated_sm, output)
output.close()

# Store results as a data frame
sm_df = create_df_from_3d_array(nn_interpolated_sm, grid_ranges, target_value=target)
file_name = 'nn_interpolated_df_' + target + '.csv'
sm_df.to_csv(storage_path + file_name, index=False)
print('Successfully stored the results...')



### Create interpolated 3D grid with Vegetation Opacity values ###
start = time.time()
target = 'vegetation_opacity'
nn_interpolated_vo = nn_interpolation(smap_pxl_df, grid_ranges, target_value=target)
print('Finished natural neighbour interpolation with vegetation_opacity as the target value in ' + str(time.time()-start) + ' seconds...')

# Store the raw array as a pickle file
output = open(storage_path + target + '_raw_array.pkl', 'wb')
pickle.dump(nn_interpolated_vo, output)
output.close()

# Store results as a data frame
sm_df = create_df_from_3d_array(nn_interpolated_vo, grid_ranges, target_value=target)
file_name = 'nn_interpolated_df_' + target + '.csv'
sm_df.to_csv(storage_path + file_name, index=False)
print('Successfully stored the results...')



### Create interpolated 3D grid with Surface Roughness values ###
start = time.time()
target = 'surface_roughness'
nn_interpolated_sr = nn_interpolation(smap_pxl_df, grid_ranges, target_value=target)
print('Finished natural neighbour interpolation with surface_roughness as the target value in ' + str(time.time()-start) + ' seconds...')

# Store the raw array as a pickle file
output = open(storage_path + target + '_raw_array.pkl', 'wb')
pickle.dump(nn_interpolated_sr, output)
output.close()

# Store results as a data frame
sm_df = create_df_from_3d_array(nn_interpolated_sr, grid_ranges, target_value=target)
file_name = 'nn_interpolated_df_' + target + '.csv'
sm_df.to_csv(storage_path + file_name, index=False)
print('Successfully stored the results...')



