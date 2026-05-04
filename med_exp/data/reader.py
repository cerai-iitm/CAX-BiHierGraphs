import pandas as pd

df = pd.read_csv('notes.csv', nrows=3)
print(df['unique_icd_codes'])
print(df.columns)
