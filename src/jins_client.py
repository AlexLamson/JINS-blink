# built to connect with and read data from JINS MEME Data Logger
# set JINS measure mode to "Full" - frame components will be in this order: MARK,NUM,DATE,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z,EOG_L,EOG_R,EOG_H,EOG_V

import sys
import socket
from datetime import datetime

s = socket.socket()

read_in_buffer = b''

# function to be called once a full frame has been recieved
on_frame = lambda x: print(frame_to_string(x))


def initialize_connection(host, port):
    s.connect( (host, port) )


def read_forever():
    read_until(lambda: False)


# reads until the condition is true
def read_until(condition_function):
    global read_in_buffer, history

    print('Looking for input...')

    while not condition_function():
        new_char = s.recv(1)
        read_in_buffer += new_char

        # if there was a newline, write the frame
        if read_in_buffer[-2:] == b'\r\n':
            new_frame_raw = read_in_buffer[:-2].decode('utf-8')
            new_frame = parse_frame(new_frame_raw)

            # frame callback function
            on_frame(new_frame)

            # reset buffer
            read_in_buffer = b''

    read_in_buffer = b''


def frame_to_string(frame):
	# turn datetime into just clock time
    result = frame[1:]
    result[1] = datetime.strftime(result[1], '%H:%M:%S.%f')[:-4]

    # if there was a mark, indicate it at the end
    if frame[0]:
        result += ['<']

    return '\t'.join(result)


def parse_frame(frame):
    elements = frame.split(',')

    # parse 'Artifact' - if there was an X, it means there's a mark on this frame
    elements[0] = elements[0] == 'X'

    # parse time
    timestring = elements[2]
    elements[2] = datetime.strptime(timestring, '%Y/%m/%d %H:%M:%S.%f')
    # could convert to unix time, or not

    return elements


if __name__ == '__main__':
    host = ''
    port = 0

    if len(sys.argv) == 3:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = input('Enter server\'s IP address: ')
        port = int( input('Enter server\'s port: ') )

    initialize_connection(host, port)

    read_forever()
