from constants import constants
import random
import pandas as pd
import shutil
import datetime


def build_data_stream():
    for size, path in zip(
        [constants["train_size"], constants["test_size"]],
        [constants["train_stream_path"], constants["test_stream_path"]],
    ):
        file = open(path, "w")
        counter = datetime.datetime.now()
        for i in range(size):
            rand_val = random.randint(0, 5)
            if rand_val < 3:
                name = "A"
                value1 = str(i)
                value2 = str(3)
                for _ in range(3):
                    s_counter = str(counter)
                    event = ",".join([name, value1, value2, value2, s_counter]) + "\n"
                    file.write(event)
                    counter += datetime.timedelta(seconds=1)
                    name = chr(ord(name) + 1)
            else:
                name = str(random.choice(constants["event_types"]))
                value1 = str(i)
                value2 = str(random.randint(0, 15))
                s_counter = str(counter)
                event = ",".join([name, value1, value2, value2, s_counter]) + "\n"
                file.write(event)
                counter += datetime.timedelta(seconds=1)
        file.close()


def pad_matches():
    def pad_matches_(is_train):
        def find_maximum_line():
            f = open(
                constants["train_matches"] if is_train else constants["test_matches"],
                "r",
            )
            max_size = 0
            i = 0
            for line in f:
                line = line.split(",")
                line[-1] = line[-1].split("\n")[0]
                this_size = len(line)
                if this_size > max_size:
                    max_size = this_size
                if i % 1000 == 0:
                    print(i)
                i += 1
            f.close()
            return max_size

        max_size = find_maximum_line()
        read_f = open(
            constants["train_matches"] if is_train else constants["test_matches"], "r"
        )
        temp_file = "temp.txt"
        write_f = open(temp_file, "w")
        i = 0
        for line in read_f:
            line = line.split(",")
            line[-1] = line[-1].split("\n")[0]
            this_size = len(line)
            for _ in range(max_size - this_size):
                line += ["-1"]
            line = ",".join(line)
            line += "\n"
            write_f.write(line)
            if i % 1000 == 0:
                print(i)
            i += 1
        read_f.close()
        write_f.close()
        shutil.move(
            temp_file,
            constants["train_matches"] if is_train else constants["test_matches"],
        )

    pad_matches_(True)
    pad_matches_(False)


def split_file(file_path, splits_num):
    read_file = open(file_path, "r")
    lines = 0
    for _ in read_file:
        lines += 1
    read_file.close()
    lines_per_split = lines / splits_num
    read_file = open(file_path, "r")
    written_lines = 0
    split_num = 1
    split_path = open(str(split_num) + ".txt", "w")
    for line in read_file:
        if written_lines != 0 and written_lines % lines_per_split == 0:
            split_path.close()
            split_num += 1
            split_path = open(str(split_num) + ".txt", "w")
        split_path.write(line)
        written_lines += 1
    split_path.close()


def combine(paths):
    t = open("file.txt", "w")
    for path in paths:
        with open(path, "r") as path:
            for line in path:
                t.write(line)

    t.close()


if __name__ == "__main__":
    print("Starting")
    build_data_stream()
    # split_file(constants['train_stream_path'], 25)
    # l = list(range(1, 26))
    # for i in range(len(l)):
    #     l[i] = str(l[i]) + ".txt"
    # combine(l)
