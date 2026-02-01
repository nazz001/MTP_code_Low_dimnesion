import pandas as pd
data=pd.read_csv("output/iris_test_pairs.csv")
data_1=pd.read_csv("output/iris_train_pairs.csv")
print(data.shape)
print(data_1.shape)