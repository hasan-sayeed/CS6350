import pandas as pd
from id3_functions import id3
from id3_functions_ensamble import predict

def bagging_model(df, n_trees, frac_sample=1):
    trees = {} ## models define
    ## set the seed to repeat the boot strapping
    for i in range(n_trees):
        #sample from X_train, y_train
        df_sampled = df.sample(frac = frac_sample, replace = True, ignore_index=True)
        trees[i] = id3(df_sampled, metric = 'entropy', tree_depth = 20)
    return(trees)

def bagging_predict(df, trees):
    y_pred = {}
    for k in trees.keys():
        train_error, y_pred[k] = predict(df, trees[k])
    # print(pd.DataFrame(y_pred))
    y_pred_list = pd.DataFrame(y_pred).mode(axis=1)[0].tolist()
    error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_pred_list) if x == y) / len(df[df.columns[-1]])
    return(error)