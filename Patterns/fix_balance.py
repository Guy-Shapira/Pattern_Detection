import pandas as pd
import csv
import numpy as np
import argparse


factor = 5
# org_file = "Patterns/pattern28_pre_final.csv"
# org_file = "Patterns/test_patterns_StarPilot.csv"
org_file = "Patterns/pattern16.csv"
dst_file = "Patterns/pattern16_fixed.csv"

df = pd.read_csv(org_file)
counts = np.zeros(10 * factor)
# out_file = open(f"Patterns/pattern28_{10 * factor}_ratings.csv", "w")
# out_file = open(f"Patterns/test_StarPilot.csv", "w")
out_file = open(dst_file, "w")
spamwriter = csv.writer(out_file)
spamwriter.writerow(df.columns)
for i, row in df.iterrows():
    rating = int(float(row['rating']) * factor) + 1
    if rating > 10 * factor:
        rating = 10 * factor
    second_max = np.sort(counts)[len(counts) // 2]
    # if counts[rating - 1] <= second_max + 1000:
    if True:
        counts[rating - 1] += 1
        row['rating'] = str(rating)
        spamwriter.writerow(row)

    if counts.sum() % 100 == 0:
        print(counts)
        input("check sanity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix balance in csv of patterns and ratings')
    parser.add_argument('--max_size', default=8, type=int, help='max size of pattern')
