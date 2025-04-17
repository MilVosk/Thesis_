import pandas as pd
df = pd.read_csv(r'C:\Users\marce\Desktop\thesis\scripts\train.csv', names=["Label", "Text"])
train = (df.drop(index=df.index[[0]]))
#print(train.head())