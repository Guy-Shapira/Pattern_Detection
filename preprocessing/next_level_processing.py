import random
import re
import numpy as np
INPUT_FILE = "store_folder/finalMatches.txt"
OUTPUT_FILE = "Football/secondLevel"


def main():
    cols = ["'x'", "'y'", "'z'", "'vx'", "'vy'"]
    mapping = {str(i) : str(i-1) for i in range(1, 11)}
    last_timestamp = {i : 0 for i in mapping.keys()}
    print(mapping)
    last_was_event = False
    next_str = ""
    with open(OUTPUT_FILE, "w+") as out_f:
        with open(INPUT_FILE, "r") as read_f:
            data = read_f.read()
            data = re.split("([0-9]|10): \{", data)
            for info in data:
                # print(info)
                if info == "":
                    continue
                elif info.startswith("'sid'"):
                    info = "{" + info
                if info.endswith("\n"):
                    info = info[:-2]
                if info in mapping:
                    if last_was_event:
                        print("Shit")
                        exit()
                    else:
                        last_was_event = True
                        next_str = mapping[info]
                else:
                    if not last_was_event:
                        print("Shit2")
                        exit()
                    else:
                        last_was_event = False
                        # print(info)
                        ts = re.search("'ts': [0-9]+", info).group()
                        ts = int(re.search("[0-9]+", ts).group())
                        event = str(int(next_str) + 1)
                        if ts > last_timestamp[event]:
                            last_timestamp[event] = ts
                            next_str += "," + str(ts)
                            for col in cols:
                                col_info = re.findall(col + ": -?[0-9]+", info)
                                col_info = np.mean([int(re.search("-?[0-9]+", i).group()) for i in col_info])
                                next_str += "," + str(int(col_info))
                            next_str += "\n"
                            out_f.write(next_str)


if __name__ == "__main__":
    main()
