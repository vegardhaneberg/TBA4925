# Standard imports
from doctest import IGNORE_EXCEPTION_DETAIL
import pandas as pd
import time

from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

# Machine Learning
import h2o
from h2o.automl import H2OAutoML

from sklearn.metrics import mean_squared_error

congo_df = pd.read_csv('/Users/madsrindal/Desktop/standard_processed_ml_df_congo.csv')
# iran_df = pd.read_csv('/Users/madsrindal/Desktop/standard_processed_ml_df_iran.csv')
# india_df = pd.read_csv('/Users/madsrindal/Desktop/standard_processed_ml_df_india.csv')

def filter_smap_day(df, start_hour, end_hour):
    filtered_df = df[df['time'] >= start_hour]
    filtered_df = filtered_df[filtered_df['time'] < end_hour]
    return filtered_df


def split_df_by_date(df, day, month, year):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    
    test_df = df[(df['day_of_year'] >= day_of_year) & (df['year'] == year)]
    test_df = test_df.append(test_df[test_df['year'] > year], ignore_index=True)
    
    training_df = pd.concat([df, test_df, test_df]).drop_duplicates(keep=False)
    
    return training_df, test_df


def split_smap_df_by_date(df, day, month, year):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    
    test_df = df[(df['time'] >= day_of_year) & (df['year'] == year)]
    test_df = test_df.append(test_df[test_df['year'] > year], ignore_index=True)
    
    training_df = pd.concat([df, test_df, test_df]).drop_duplicates(keep=False)
    
    return training_df, test_df


def filter_location_smap(df, location):
    filtered_df = df[df.lat < location[0]]
    filtered_df = filtered_df[filtered_df.lat > location[2]]
    filtered_df = filtered_df[filtered_df.long < location[3]]
    filtered_df = filtered_df[filtered_df.long > location[1]]
    return filtered_df


def round_nearest(x, a):
    return round(x / a) * a


def get_dem_parameters(lat, long, dem_df):
    lst = list(dem_df['rounded_long'].unique())[:2]
    grid_box_size = round(lst[1] - lst[0], 5)
    rounded_lat = round(round_nearest(lat, grid_box_size), 5)
    rounded_long = round(round_nearest(long, grid_box_size), 5)
    
    tmp_df = dem_df[(dem_df['rounded_lat'] == rounded_lat) & (dem_df['rounded_long'] == rounded_long)]
    row = tmp_df.iloc[0]
    return row['delta_angle'], row['rmse_distance']


def set_dem_parameters(cygnss_df, dem_df):
    cygnss_df['delta_angle'] = cygnss_df.progress_apply(lambda row: get_dem_parameters(row.sp_lat, row.sp_lon, dem_df)[0], axis=1)
    cygnss_df['rmse_distance'] = cygnss_df.progress_apply(lambda row: get_dem_parameters(row.sp_lat, row.sp_lon, dem_df)[1], axis=1)
    return cygnss_df


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


def filter_inc_angles(cygnss_df, min_angle, max_angle):
    cygnss_df = cygnss_df[(cygnss_df['sp_inc_angle'] >= min_angle) & (cygnss_df['sp_inc_angle'] <= max_angle)]
    return cygnss_df


# Returning the same df if no specific features are selected
def select_df_features(df, feature_list):
    if len(feature_list) > 0:
        return df[feature_list]
    else:
        return df


# CONGO
# pixel_loc = [-10, 27, -10.5, 27.5] # Best cell
# pixel_loc = [-8, 27.5, -8.5, 28] # Worst cell
pixel_loc = [-10, 23.5, -10.5, 24] # Forest cell
congo_df = filter_location(congo_df, pixel_loc)

"""tmp_df = filter_inc_angles(congo_df, 11, 21)
congo_df = tmp_df.append(filter_inc_angles(congo_df, 45, 55), ignore_index=True)

print('Max Inc Angle: ', congo_df['sp_inc_angle'].max())
print('Min Inc Angle: ', congo_df['sp_inc_angle'].min())

#dem_df = pd.read_csv('/Users/madsrindal/Desktop/DEM LUTs/Congo/LUT_Congo_BestCorrCell_0.01.csv') # BestCell
dem_df = pd.read_csv('/Users/madsrindal/Desktop/DEM LUTs/Congo/LUT_Congo_WorstCorrCell_0.01.csv') # WorstCell
#dem_df = dem_df.append(pd.read_csv('/Users/madsrindal/Desktop/DEM LUTs/Congo/LUT_Congo_ForestCorrCell_0.01.csv'), ignore_index=True) # ForestCell
congo_df = set_dem_parameters(congo_df, dem_df)"""

"""# INDIA
# pixel_loc = [21.5, 81.5, 21, 82] # Best cell
pixel_loc = [20, 83.5, 19.5, 84] # Worst cell
india_df = filter_location(india_df, pixel_loc)"""

"""# IRAN
pixel_loc = [27.5, 59, 27, 59.5] # Best cell
# pixel_loc = [30.5, 59.5, 30, 60] # Worst cell
iran_df = filter_location(iran_df, pixel_loc)"""


# Create train/test split in dataset.
# The value on format [day, month, year] indicates test data from provided date 
data_split_date = [1, 7, 2021]

ml_df = congo_df.copy()
ml_features = ['sr', 'smap_sm']
# ml_features = ['sr', 'smap_sm', 'day_of_year', 'sp_inc_angle', 'smap_vo', 'smap_surface_roughness', 'block_code', 'time_of_day']
# ml_features = ['sr', 'smap_sm', 'day_of_year', 'sp_inc_angle', 'smap_vo', 'smap_surface_roughness', 'block_code', 'time_of_day', 'delta_angle', 'rmse_distance']

ml_df_train, ml_df_test = split_df_by_date(ml_df, data_split_date[0], data_split_date[1], data_split_date[2])
ml_df_train = select_df_features(ml_df_train, ml_features)
ml_df_test = select_df_features(ml_df_test, ml_features)

for i in range(2):
    try:
        h2o.cluster().shutdown()
    except:
        pass

h2o.init()

train_h2o = h2o.H2OFrame(ml_df_train)
test_h2o = h2o.H2OFrame(ml_df_test)

y = 'smap_sm'
x = train_h2o.columns
x.remove(y)

#train_h2o['day_of_year'] = train_h2o['day_of_year'].asfactor()
#train_h2o['block_code'] = train_h2o['block_code'].asfactor()
#train_h2o['time_of_day'] = train_h2o['time_of_day'].asfactor()

#test_h2o['day_of_year'] = test_h2o['day_of_year'].asfactor()
#test_h2o['block_code'] = test_h2o['block_code'].asfactor()
#test_h2o['time_of_day'] = test_h2o['time_of_day'].asfactor()

start_time = time.time()
aml = H2OAutoML(balance_classes=False, max_models = 15, seed = 1, include_algos = ["xgboost", "GBM", "DRF", "StackedEnsemble", "DeepLearning"])
aml.train(x = x, y = y, training_frame = train_h2o)
print('Finished the training process in a total of ' + str(time.time()-start_time) + ' seconds...\n')
    
model1 = aml.get_best_model(algorithm="xgboost", criterion="rmse")
model2 = aml.get_best_model(algorithm="GBM", criterion="rmse")
model3 = aml.get_best_model(algorithm="DRF", criterion="rmse")
model4 = aml.get_best_model(algorithm="StackedEnsemble", criterion="rmse")
model5 = aml.get_best_model(algorithm="DeepLearning", criterion="rmse")

model_name1 = 'xgboost_rmse_' + str(ml_features)
model_name2 = 'GBM_rmse_' + str(ml_features)
model_name3 = 'DRF_rmse_' + str(ml_features)
model_name4 = 'StackedEnsemble_rmse_' + str(ml_features)
model_name5 = 'DeepLearning_rmse_' + str(ml_features)

model_path1 = h2o.save_model(model=model1, path="/tmp/mymodel", force=True)
model_path2 = h2o.save_model(model=model2, path="/tmp/mymodel", force=True)
model_path3 = h2o.save_model(model=model3, path="/tmp/mymodel", force=True)
model_path4 = h2o.save_model(model=model4, path="/tmp/mymodel", force=True)
model_path5 = h2o.save_model(model=model5, path="/tmp/mymodel", force=True)

"""model1_local1 = h2o.download_model(model1, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/WorstCell/optimal_inc/10features/both_ranges") # + str(ml_features))
model1_local2 = h2o.download_model(model2, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/WorstCell/optimal_inc/10features/both_ranges") # + str(ml_features))
model1_local3 = h2o.download_model(model3, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/WorstCell/optimal_inc/10features/both_ranges") # + str(ml_features))
model1_local4 = h2o.download_model(model4, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/WorstCell/optimal_inc/10features/both_ranges") # + str(ml_features))
model1_local5 = h2o.download_model(model5, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/WorstCell/optimal_inc/10features/both_ranges") # + str(ml_features))"""

model1_local1 = h2o.download_model(model1, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/ForestCell/2features") # + str(ml_features))
model1_local2 = h2o.download_model(model2, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/ForestCell/2features") # + str(ml_features))
model1_local3 = h2o.download_model(model3, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/ForestCell/2features") # + str(ml_features))
model1_local4 = h2o.download_model(model4, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/ForestCell/2features") # + str(ml_features))
model1_local5 = h2o.download_model(model5, path="/Users/madsrindal/Desktop/AutoML - Models/Congo/ForestCell/2features") # + str(ml_features))


preds1 = model1.predict(test_h2o)
preds2 = model2.predict(test_h2o)
preds3 = model3.predict(test_h2o)
preds4 = model4.predict(test_h2o)
preds5 = model5.predict(test_h2o)

predictions1 = h2o.as_list(preds1)
predictions2 = h2o.as_list(preds2)
predictions3 = h2o.as_list(preds3)
predictions4 = h2o.as_list(preds4)
predictions5 = h2o.as_list(preds5)

rmse_xgboost = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions1['predict'], squared=False)
rmse_gbm = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions2['predict'], squared=False)
rmse_drf = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions3['predict'], squared=False)
rmse_stack = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions4['predict'], squared=False)
rmse_dl = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions5['predict'], squared=False)

rmse_df = pd.DataFrame()
rmse_df['xgboost'] = [rmse_xgboost]
rmse_df['gbm'] = [rmse_gbm]
rmse_df['drf'] = [rmse_drf]
rmse_df['stack'] = [rmse_stack]
rmse_df['dl'] = [rmse_dl]
# rmse_df.to_csv('/Users/madsrindal/Desktop/Congo_rmse_df_Concated_' + str(len(ml_features)) + 'features_small_gridbox_optimal_inc[11-21].csv', index=False)
#rmse_df.to_csv('/Users/madsrindal/Desktop/Congo_rmse_df_WorstCell_' + str(len(ml_features)) + 'features_small_gridbox_both_optimal_inc.csv', index=False)
rmse_df.to_csv('/Users/madsrindal/Desktop/Congo_rmse_df_ForestCell_' + str(len(ml_features)) + 'features.csv', index=False)
