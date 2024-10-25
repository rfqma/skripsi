import pandas as pd
import glob

csv_files = glob.glob('datasets/raw/*.csv')
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df.to_csv('datasets/merged/merged_dataset.csv')