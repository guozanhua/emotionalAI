from glob import glob
import pandas as pd

seqs = glob('./data/sequences/*')
for i in seqs:
    print(i)
    pd.read_csv(i, sep=" ", header=None)
