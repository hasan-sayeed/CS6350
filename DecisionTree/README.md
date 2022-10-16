The decision tree algorithm can be used by the following line of code:

`id3(training_data, metric, tree_depth)`

Here, `metric` can be `'entropy'`, `'majority error'` or `'gini index'` and `tree_depth` is the maximum depth of the tree user wants.

To predict the accuracy of the tree on a dataset following function can be called:

`predict(test_data, tree)`

Where `test_data` is the dataset user wants to use the previously constructed tree on and `tree` is the tree that can be constructed following the previous step.