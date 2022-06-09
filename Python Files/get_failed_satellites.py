import pandas as pd
from calendar import monthrange, month_name
from tqdm import tqdm
tqdm.pandas()


def find_missing_value(value_list):
    return list(set(range(1, 9)) - set(value_list))


def get_failed_satellites_month(year, month):
    num_of_days = monthrange(year, month)[1]
    retrieval_folder = '/Volumes/DACOTA HDD/Semester Project CSV/CYGNSS ' + str(year) + '-' + str(month).zfill(2)
    
    failed_list = [] #format: [ [day, [sats]] ]

    for i in tqdm(range(num_of_days)):
        csv_path = retrieval_folder + '/raw_main_df_' + str(year) + '_' + str(month).zfill(2) + '_' + \
                   str(i + 1) + 'of' + str(num_of_days) + '.csv'
        daily_df = pd.read_csv(csv_path)
        found_sats = list(daily_df['spacecraft_num'].unique())
        missing_sats = find_missing_value(found_sats)
        
        if len(missing_sats) > 0:
            failed_list.append([i + 1, missing_sats])
        
    return failed_list


### Get failed satellites for an entire year ###
year = 2021

months = []
sats = []

for month in range(1, 13):
    failed_sats = get_failed_satellites_month(year, month)
    months.append(month)
    sats.append(failed_sats)

failed_sat_df = pd.DataFrame()
failed_sat_df['month'] = months
failed_sat_df['missing_satellites'] = sats

failed_sat_df.to_csv('/Users/madsrindal/Desktop/all_failed_satellites_' + str(year) + '.csv', index=False)