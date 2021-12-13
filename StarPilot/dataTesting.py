import os


ORIG_FILE = "output1.txt"
REDUCED_FILE = "StrpilotData1.txt"
FINAL_FOLDER = "GamesExp/"


def phase1():
    file1 = open(ORIG_FILE)
    file2 = open(REDUCED_FILE, "a")
    count = 0
    curr_lines = []
    for line in file1.readlines():
        curr_lines.append(line)
        if "---" not in line:
            count += 1
        else:
            if count >= 500:
                file2.writelines(curr_lines)
            curr_lines = []
            print(count)
            count = 0

def phase2():
    counter = 0
    file1 = open(REDUCED_FILE)
    if not os.path.exists(FINAL_FOLDER):
        os.mkdir(FINAL_FOLDER)
    curr_lines = []
    for line in file1.readlines():
        if not "---" in line:
            curr_lines.append(line)
        else:
            new_file = open(FINAL_FOLDER + f"data_{str(counter)}.txt", "a")
            new_file.writelines(curr_lines)
            curr_lines = []
            counter += 1

if __name__ == "__main__":
    phase1()
    phase2()
