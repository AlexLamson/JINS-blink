# connects to jins and shows the model's evaluation in realtime

import pickle
import sys
import jins_client
import train
import numpy as np
from time import time

window_size = 2

window = np.zeros((window_size, 4))

model = None
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

update_interval = 0.25
last_draw_time = 0
blinks_this_interval = 0
blink_threshold = 5


def frame_to_windowable(frame):
    result = [ int( frame[i] ) for i in [9, 10, 11, 12] ]
    return np.array(result)


def classify(frame):
    global window, model, update_interval, last_draw_time, blinks_this_interval, blink_threshold

    window[0] = window[1]
    window[1] = frame_to_windowable(frame)

    features = np.array([ train.extract_features(window) ])

    prediction = model.predict(features)

    if prediction:
    	blinks_this_interval += 1

    if time() - last_draw_time > update_interval:
    	print( '\tblink' if blinks_this_interval >= blink_threshold else 'open' )

    	last_draw_time = time()
    	blinks_this_interval = 0


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
