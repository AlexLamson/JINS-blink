# trains a model to identify blinks
from util import *
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


def extract_features(window):
    # in: n frames of EOG L, R, H, V

    # window -= np.mean(window, axis=0)  # shift the baseline sometime

    window_mins = np.min(window, axis=0)
    window_maxs = np.max(window, axis=0)

    out = np.mean(window, axis=0)  # mean of each EOG value
    out = np.append( out, window_mins )  # min for each EOG value
    out = np.append( out, window_maxs )  # max for each EOG value
    out = np.append( out, window_maxs - window_mins )  # max - min for each EOG value
    out = np.append( out, window[1] - window[0] )  # diff from start to end for each EOG value
    out = np.append( out, np.percentile(window, q=25, axis=0) )
    out = np.append( out, np.percentile(window, q=50, axis=0) )
    out = np.append( out, np.percentile(window, q=75, axis=0) )

    magnitudes = np.linalg.norm(window, axis=1)  # magnitude of each frame

    out = np.append( out, np.mean(magnitudes) )
    out = np.append( out, np.min(magnitudes) )
    out = np.append( out, np.max(magnitudes) )
    out = np.append( out, out[-1] - out[-2] )  # max - min of magnitudes
    out = np.append( out, magnitudes[-1] - magnitudes[0] )  # diff from start to end magnitude

    # out = np.append( out, np.histogram(window[:,3], normed=True) )

    return out


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


def slidingWindow(sequence, winSize, stride=1):
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
    if not (isinstance(winSize, int) and isinstance(stride, int)):
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


def param_optimization(model, params):
    RandomizedSearchCV()


    import numpy as np
    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.grid_search import RandomizedSearchCV

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf1 = DecisionTreeClassifier()
    clf2 = RandomForestClassifier(random_state=rand_seed)

    params = {'dt__max_depth': [5, 10], 'rf__n_estimators': [20, 200],}


    eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2)], voting='hard')

    random_search = RandomizedSearchCV(eclf, param_distributions=params,n_iter=4)
    random_search.fit(X, y)
    print(random_search.grid_scores_)


def make_roc_curve(truth, model_prediction_scores):
    pass


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


    # subsample data so the models train faster
    print("sampling (time-based) less data for faster training")
    seconds_of_data = 60*30
    inputs, outputs = inputs[:seconds_of_data*100], outputs[:seconds_of_data*100]


    # sample blinks more often than open eyes to address class imbalance
    def should_sample(OF_blink_confidence):
        if OF_blink_confidence < 0.025:
            return random() < 0.11
        return True
    print("sampling (confidence-based) less data for faster training")
    samples_mask = np.array([should_sample(OF_blink_confidence) for OF_blink_confidence in outputs])
    inputs = inputs[samples_mask]
    outputs = outputs[samples_mask]


    # train the model
    """
    Regression models
    """
    # model = LinearRegression()
    # model = DecisionTreeRegressor(max_depth=3)
    # evaluate_model(model, inputs, outputs, is_classification=False)

    """
    Classification models
    """
    threshold = 0.6
    classification_outputs = outputs >= threshold

    # results = []
    # threshold = 0.6
    # threshold_grid = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # for threshold in threshold_grid:
    #     # model = LogisticRegression()
    #     model = RandomForestClassifier(n_estimators=4,max_depth=8,min_samples_split=10)
    #     # model = DecisionTreeClassifier(max_depth=3)
    #     # model = LinearSVC()
    #     # model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    #     result = evaluate_model(model, inputs, classification_outputs, is_classification=True)
    #     results.append(result)

    # # plot the grid search results
    # results = np.array(results)
    # results_nonblink = results[:,0]
    # results_blink = results[:,0]

    # # print("A size: {} B size: {}".format(len(threshold_grid), len(results)))
    # plt.scatter(x=threshold_grid, y=results_blink, c='r')
    # plt.show()
    # plt.scatter(x=threshold_grid, y=results_nonblink, c='b')
    # plt.show()






    # results = []
    # # model = LogisticRegression()
    # model = RandomForestClassifier(n_estimators=4,max_depth=8,min_samples_split=10)
    # # model = DecisionTreeClassifier(max_depth=3)
    # # model = LinearSVC()
    # # model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # # evaluate_model(model, inputs, classification_outputs, is_classification=True)

    # model.fit(inputs, classification_outputs)
    # # print("lovey shape: {}".format(classification_outputs.shape))
    # # print("dovey shape: {}".format(model.predict_proba(inputs).shape))

    # # np.savetxt('corgi.txt', model.predict_proba(inputs)[:,1])
    # # exit()

    # fpr, tpr, _ = roc_curve(classification_outputs, model.predict_proba(inputs)[:,1])

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange',
    #          label='ROC curve (area = ???)')
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # plt.legend(loc="lower right")
    # plt.show()





    model = RandomForestClassifier(n_estimators=20,min_samples_split=4,min_samples_leaf=2)

    # X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2) 
    # y_train, y_test = y_train >= threshold , y_test >= threshold


    # plt.figure()

    # for hyperparam in [2, 4, 10, 20, 50, 100, 500]:
    #     print("testing with hyperparam set to {}".format(hyperparam))
    #     model = RandomForestClassifier(n_estimators=20,min_samples_split=4,min_samples_leaf=hyperparam)
    #     model.fit(X_train, y_train)
    #     fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    #     plt.plot(fpr, tpr, label='{} (area = {})'.format(hyperparam, '???'))

    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # plt.legend(loc="lower right")
    # plt.show()




    # save the model
    print('fitting whole dataset')
    model.fit(inputs, classification_outputs)  # be sure to re-train the best model on the full data-set
    save_obj(model, 'model.pickle')
