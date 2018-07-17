'''
NOTE: all this code is a placeholder for now.
The classifier evaluator will be made before this one,
and this should follow whatever patterns are created in there.
'''
def evaluate_model(model, inputs, outputs, is_classification=True):
    if is_classification:
        print("evaluating classification model")
    else:
        print("evaluating regression model")

    cv = KFold(n_splits=10, shuffle=True, random_state=rand_seed)

    # used for averaging the confusion matrices, fscores
    conf_array = []
    f_array = []
    loss_array = []

    for i, (train_indices, test_indices) in enumerate(cv.split(inputs)):
        # split into training and testing
        inputs_train = inputs[train_indices, :]
        outputs_train = outputs[train_indices]
        inputs_test = inputs[test_indices, :]
        outputs_test = outputs[test_indices]

        model.fit(inputs_train, outputs_train)

        predictions = model.predict(inputs_test)

        if is_classification:
            conf_array += [ confusion_matrix(outputs_test, predictions) ]
            _, _, fscore, _ = precision_recall_fscore_support(outputs_test, predictions)
            f_array += [ fscore ]
        else:
            l1_loss = abs(outputs_test-predictions)
            loss_array += [ np.mean(l1_loss) ]

    if is_classification:
        print('predicted\n open\tblink')
        print( np.mean(conf_array, axis=0) )
        fscore = np.mean(f_array, axis=0)
        print('fscore', fscore)
        return fscore
    else:
        avg_l1_loss = np.mean(l1_loss)
        print("average L1 loss: {}".format( avg_l1_loss ))
        return avg_l1_loss
