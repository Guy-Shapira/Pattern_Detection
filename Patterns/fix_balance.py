import pandas as pd
import csv
import numpy as np

factor = 5
# org_file = "Patterns/pattern28_pre_final.csv"
org_file = "Patterns/test_patterns_StarPilot.csv"

df = pd.read_csv(org_file)
counts = np.zeros(10 * factor)
# out_file = open(f"Patterns/pattern28_{10 * factor}_ratings.csv", "w")
out_file = open(f"Patterns/test_StarPilot.csv", "w")
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
