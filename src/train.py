# trains a model to identify blinks
from util import *
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def extract_features(window):
    # in: n frames of EOG L, R, H, V

    out = np.mean(window, axis=0)  # mean of each EOG value
    out = np.append( out, np.min(window, axis=0) )  # min for each EOG value
    out = np.append( out, np.max(window, axis=0) )  # max for each EOG value
    out = np.append( out, np.max(window, axis=0) - np.min(window, axis=0) )  # max - in for each EOG value
    out = np.append( out, window[1] - window[0] )  # diff from start to end for each EOG value

    magnitudes = np.linalg.norm(window, axis=1)  # magnitude of each frame

    out = np.append( out, np.mean(magnitudes) )
    out = np.append( out, np.min(magnitudes) )
    out = np.append( out, np.max(magnitudes) )
    out = np.append( out, out[-1] - out[-2] )  # max - min of magnitudes
    out = np.append( out, magnitudes[-1] - magnitudes[0] )  # diff from start to end magnitude

    return out


def evaluate_model(model, inputs, outputs, is_classification=True):
    if is_classification:
        print("evaluating classification model")
    else:
        print("evaluating regression model")

    cv = KFold(n_splits=10, shuffle=True, random_state=None)

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
        print('fscore', np.mean(f_array, axis=0))
    else:
        print("average L1 loss: {}".format( np.mean(l1_loss) ))


def slidingWindow(sequence,winSize,stride=1):
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
    if not ((type(winSize) == type(0)) and (type(stride) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(stride) must be int.")
    if stride > winSize:
        raise Exception("**ERROR** stride must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence)-winSize)/stride)+1

    # Do the work
    for i in range(0,numOfChunks*stride,stride):
        yield i, sequence[i:i+winSize]


# put the data in terms of inputs and outputs
def extract_features_from_dataframe(df, window_size):
    data = df.as_matrix()  # turn it into a numpy array

    inputs = []
    outputs = []

    samples = df.shape[0] - window_size + 1

    with tqdm(total=samples) as pbar:
        for i, window_full in slidingWindow(data, window_size):
            window = window_full[:, 2:-1]  # shave off frame id, time, and blink value
            features = extract_features(window)

            inputs += [ features ]
            # outputs += [ np.mean(window_full[:, -1]) >= 0.6 ]  # turn the blink data into booleans
            outputs += [ window_full[int(window_full.shape[0]/2), -1] ]  # leave the blink regression as-is

            pbar.update(1)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


if __name__ == '__main__':
    # training_data_fname = '../res/data4/combined.csv'
    training_data_fname = data_folder+'combined.csv'

    window_size = 10

    print('loading data')
    data = pd.read_csv(training_data_fname)
    print('done loading')

    features_pickle_filename = data_folder+"features.pickle"
    if file_exists(features_pickle_filename):
        print('loading precomputed features')
        inputs, outputs = load_obj(features_pickle_filename)
    else:
        print('extracting features & saving to file')
        inputs, outputs = extract_features_from_dataframe(data, window_size)
        save_obj((inputs, outputs), features_pickle_filename)


    # uncomment line to sample it so the models train faster
    seconds_of_data = 60*10
    inputs, outputs = inputs[:seconds_of_data*100], outputs[:seconds_of_data*100]


    # train the model
    """
    Regression models
    """
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # from sklearn.tree import DecisionTreeRegressor
    # model = DecisionTreeRegressor(max_depth=3)
    # evaluate_model(model, inputs, outputs, False)

    """
    Classification models
    """
    outputs = outputs >= 0.6
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression()
    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier(max_depth=3)
    # from sklearn.svm import LinearSVC
    # model = LinearSVC()
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    evaluate_model(model, inputs, outputs)

    # save the model
    print('fitting whole dataset')
    model.fit(inputs, outputs)  # be sure to re-train the best model on the full data-set
    save_obj(model, 'model.pickle')
