import pandas as pd
from functions import construct_tree


df_train = pd.read_csv('data/car/train.csv')
tree = construct_tree(df_train)
import pprint
pprint.pprint(tree)