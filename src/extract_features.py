from tqdm import tqdm
import numpy as np


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
