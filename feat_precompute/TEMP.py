import pandas as pd
# pred = pd.read_csv("/project/rrg-josedolz/natgill/ShareLock_hist/feat_precompute/predictions_quiltcleaner.csv.bz2")   # pandas reads .bz2 directly
# print(pred.columns, pred.head())         

quilt = pd.read_csv("/scratch/natgill/quilt1m/quilt_1M_lookup.csv")
print(quilt.columns, quilt.head())
