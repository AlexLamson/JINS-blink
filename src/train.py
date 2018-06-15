# trains a model to identify blinks

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

def extract_features(window):
    # in: n frames of EOG L, R, H, V

    out = np.mean(window, axis=0)  # mean of each EOG value
    out = np.append( out, window[1] - window[0] )  # diff from start to end of each EOG value

    return out

def evaluate_model(model, inputs, outputs):
    n = len(inputs)

    cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

    for i, (train_indices, test_indices) in enumerate(cv):
        # split into training and testing
        inputs_train = inputs[train_indices, :]
        outputs_train = outputs[train_indices]
        inputs_test = inputs[test_indices, :]
        outputs_test = outputs[test_indices]

        model.fit(inputs_train, outputs_train)

        predictions = model.predict(inputs_test)

        tp = fp = tn = fn = 0
        for i, prediction in enumerate(predictions):
            actual = outputs_test[i]

            if prediction == actual:
                if actual == False:
                    tn += 1
                else: tp += 1
            else:
                if actual:
                    fn += 1
                else:
                    fp += 1

        print('tp', tp)
        print('tn', tn)
        print('fp', fp)
        print('fn', fn)
        print('correct\t', (tp+tn) / len(predictions))
        print('wrong\t', (fp+fn) / len(predictions))
        print('precision?\t', (tp/(tp+fn)))
        print()

        # print( confusion_matrix(outputs_test, predictions, labels = ['no blink', 'blink']) )

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    @author: snoran
    Thanks to https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/"""
 
    # Verify the inputs
    try: 
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence)-winSize)/step)+1
    
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield i, sequence[i:i+winSize]

if __name__ == '__main__':
    training_data_fname = '../res/data1/combined.csv'

    print('loading data')
    data = pd.read_csv(training_data_fname).as_matrix()  # read in data and turn it into a numpy array
    print('done loading')

    # put the data in terms of inputs and outputs
    print('extracting features')
    window_size = 2

    inputs = []
    outputs = []

    for i, window_full in slidingWindow(data, window_size):
        window = window_full[:, 2:-1]  # shave off frame id, time, and blink value
        features = extract_features(window)

        inputs += [ features ]
        outputs += [ np.mean(window_full[:, -1]) >= 0.4 ]  # turn the blink data into booleans

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # train the model
    print('evaluating')

    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    # from sklearn import svm
    # model = svm.SVC()
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    evaluate_model(model, inputs, outputs)

    # save the model
    # be sure to re-train the best model on the full data-set
