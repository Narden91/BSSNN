
import os

# Create dummy data if not exists
if not os.path.exists('./data/ETT/'):
    os.makedirs('./data/ETT/')

# We need a real file to test, or we can mock it. 
# Since I cannot download files, I will assume the user has the data OR I will create a dummy csv.
# Creating a dummy ETTh1.csv for testing purposes
import pandas as pd
import numpy as np

if not os.path.exists('./data/ETT/ETTh1.csv'):
    print("Creating dummy ETTh1.csv for sanity check...")
    dates = pd.date_range(start='2016-07-01', periods=1000, freq='H')
    data = np.random.randn(1000, 7)
    df = pd.DataFrame(data, columns=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])
    df['date'] = dates
    # move date to first col
    cols = ['date'] + [c for c in df.columns if c != 'date']
    df = df[cols]
    df.to_csv('./data/ETT/ETTh1.csv', index=False)

# Run the command
cmd = "python experiments/run_benchmark.py --data ETTh1 --train_epochs 2 --seq_len 24 --pred_len 24 --patience 1 --batch_size 16"
print(f"Running: {cmd}")
os.system(cmd)
