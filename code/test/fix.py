import pandas as pd
df = pd.read_csv('./annotation.csv')
# 假设第一列是图片名
df.iloc[:, 0] = df.iloc[:, 0].astype(str) + '.jpg' 
df.to_csv('./annotation_fixed.csv', index=False)