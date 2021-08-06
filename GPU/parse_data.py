import pandas as pd
from datetime import datetime, time, timedelta
import typing


LIST_OF_SERVERS = [2, 3, 4, 5, 6]
# LIST_OF_SERVERS = [2, 3]
START_HOUR_CHAR = 11
END_HOUR_CHAR = 12
LIST_OF_FILES = [f"GPU/data_cleaned/gpuData_rambo{i}.csv" for i in LIST_OF_SERVERS]
FORMAT_STR = '%a %b %d %H %M %S %Y'

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
        # print(e)
        # raise(e)
        pass


def main():
    main_frame = None
    for csv_file, server_number in zip(LIST_OF_FILES, LIST_OF_SERVERS):
        df = pd.read_csv(csv_file, header=0, index_col=0)
        df = df.drop(df[df.Timestamp ==  'Driver/libraryversionmismatch'].index)
        old_first = df['Power Samples Avg GPU_4'][0]
        # print(df.groupby('Timestamp')['Timestamp'].nunique())
        # exit(0)
        df['Timestamp'] = df['Timestamp'].apply(get_str_obj)
        df = df.groupby(by=["Timestamp"]).mean()
        new_first = df['Power Samples Avg GPU_4'][0]
        if False:
            assert (old_first != new_first)
        
        df['Server'] = server_number

        # df['tmp'] = 1
        print(df.head())
        if main_frame is None:
            main_frame = df
        else:
            main_frame = main_frame.append(df)
    
    main_frame = main_frame.sort_values(by=['Timestamp'])
    print(main_frame.head())
    main_frame.to_csv(path_or_buf='GPU/data.csv')
if __name__ == "__main__":
    main()