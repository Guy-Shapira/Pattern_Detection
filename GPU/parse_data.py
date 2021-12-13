import pandas as pd
from datetime import datetime, time, timedelta
import typing
import gc


LIST_OF_SERVERS = [2, 3, 4, 5, 6]
START_HOUR_CHAR = 11
END_HOUR_CHAR = 12
LIST_OF_FILES = [f"GPU/data_cleaned/gpuData_rambo{i}.csv" for i in LIST_OF_SERVERS]
FORMAT_STR = '%a %b %d %H %M %S %Y'
DATA_PATH = 'GPU/data.csv'
NEW_DATA_PATH = 'GPU/full_data_check.csv'
FIX_DATA_PATH = 'GPU/sanity2.csv'
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_str_obj(time_str: str):
    try:
        time_str = time_str.strip('\n')
        if time_str.startswith(" "):
            return datetime.strptime(time_str[1:], FORMAT_STR)
        else:
            return datetime.strptime(time_str, FORMAT_STR)
    except Exception as e:        
        print(time_str)
        exit(0)


def main():
    main_frame = None
    for csv_file, server_number in zip(LIST_OF_FILES, LIST_OF_SERVERS):
        print(f"Starting work on rambo {server_number}")
        df = pd.read_csv(csv_file, header=0, index_col=0)
        df = df.drop(df[df.Timestamp ==  'Driver/libraryversionmismatch'].index)
        # old_first = df['Power Samples Avg GPU_4'][0]
        df['Timestamp'] = df['Timestamp'].apply(get_str_obj)
        df = df.groupby(by=["Timestamp"]).mean()
        df['Server'] = server_number
        if main_frame is None:
            main_frame = df
        else:
            main_frame = main_frame.append(df)
        print(f"Finished work on rambo {server_number}")
        
    
    main_frame = main_frame.sort_values(by=['Timestamp'])
    print(main_frame.head())
    main_frame.to_csv(path_or_buf=DATA_PATH)

# def filter_events(_25_per=None, _75_per=None, cols=None, diff=1.5):
def filter_events():
    df = pd.read_csv(DATA_PATH, header=0, index_col=False)
    last_times = {}
    columns = df.columns
    cols = list(columns)
    cols.remove('Timestamp')
    cols.remove('Server')
    _75_per = [df[col].describe()["75%"] for col in cols]
    _25_per = [df[col].describe()["25%"] for col in cols]

    kept_data = []
    avg_low, avg_high = 0,0
    print("Num cols:", len(cols))

    for index, row in df.iterrows():
        to_keep = False
        row_timestamp = datetime.strptime(row['Timestamp'], TIME_FORMAT)
        row_server = str(row['Server'])
        if not _25_per is None:
            num_lower = sum([row[col] < _25_per[i]  for i, col in enumerate(cols)])
            num_higher = sum([row[col] > _75_per[i] for i, col in enumerate(cols)])
            if num_lower >= len(cols) / 3 - 1:
                row_server += "_low"
            elif num_higher >= len(cols) / 2 - 1:
                row_server += "_high"
            else:
                row_server += "_reg"
            
            if index % 10000 == 0:
                print(f"Index: {index}, low: {num_lower}, high: {num_higher}")

            avg_low += num_lower
            avg_high += num_higher
        if last_times.get(row_server) is None:
            last_times[row_server] = row_timestamp
            to_keep = True
        else:
            last_time = last_times[row_server]
            if row_timestamp > last_time +  timedelta(seconds=2):
                last_times[row_server] = row_timestamp
                to_keep = True

        if to_keep:
            row['Server'] = row_server
            kept_data.append(row)
        
        if index % 10000 == 0:
            print(f"current index: {index}, kept: {len(kept_data)}")
            print(f"avg_low : {avg_low / (index + 1)}, avg_high: {avg_high/ (index + 1)}")
        if len(kept_data) >= 250000:
            break
    new_df = pd.DataFrame(kept_data, columns=columns)
    new_df.to_csv(path_or_buf=NEW_DATA_PATH)

def remove_counter():
    df = pd.read_csv(NEW_DATA_PATH, header=0, index_col=0)
    df.to_csv(path_or_buf=FIX_DATA_PATH, index=False)


def get_info():
    df = pd.read_csv(DATA_PATH, header=0, index_col=False)
    df = df.drop(["Timestamp", "Server"], axis=1)
    # last_times = {}
    columns = df.columns
    _75_per = [df[col].describe()["75%"] for col in columns]
    _25_per = [df[col].describe()["25%"] for col in columns]
    del df
    gc.collect()
    return _25_per, _75_per, columns



if __name__ == "__main__":
    # main()
    # _25_per, _75_per, cols = get_info()
    # print(cols)
    # filter_events(_25_per, _75_per, cols, diff=1.5)
    filter_events()
    remove_counter()


