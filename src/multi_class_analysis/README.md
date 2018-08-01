How to use
----------

To change the data path, change the string on line 15 of `prepare_data.py`.


To train the model & view the conf matrix, open `train_machine_learning_model.py` and change line 54 `use_precomputed=True` to `use_precomputed=False`. Then run the file.

The other filenames should be fairly self explanatory.


File descriptions
-----------------

* clustering_visualizations.py - Run PCA, t-SNE, etc to visualize data. Will take a minute or two to run.
* feature_extractor.py - Contains all the code to compute the features.
* prepare_data.py - Script to load the data into other scripts.
* train_machine_learning_model.py - Trains a model and prints some info & plots the confusion matrix.
* util.py - Some general utilities not related to machine learning.
* view_example_window.py - Plots each of the 10 (normalized) signals in one graph.
* visualize_decision_tree.py - Plots a decision tree so you can see what features it likes to use.
