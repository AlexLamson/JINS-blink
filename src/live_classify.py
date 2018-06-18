# connects to jins and shows the model's evaluation in realtime

import pickle
import sys
import jins_client
import train
import numpy as np


window = np.zeros((2, 4))

model = None
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


def frame_to_windowable(frame):
    result = [ int( frame[i] ) for i in [9, 10, 11, 12] ]
    return np.array(result)


def classify(frame):
    global window, model

    window[0] = window[1]
    window[1] = frame_to_windowable(frame)

    features = np.array([ train.extract_features(window) ])

    prediction = model.predict(features)

    print( '\tblink' if prediction else 'open' )


if __name__ == '__main__':
    host = ''
    port = 0

    if len(sys.argv) == 3:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = input('Enter server\'s IP address: ')
        port = int( input('Enter server\'s port: ') )

    jins_client.initialize_connection(host, port)

    jins_client.on_frame = classify

    jins_client.read_forever()
