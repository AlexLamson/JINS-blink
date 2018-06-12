# built to connect with and read data from JINS MEME Data Logger
# set JINS measure mode to "Full" - frame components will be in this order: NUM,DATE,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z,EOG_L,EOG_R,EOG_H,EOG_V

import sys
import socket
from datetime import datetime

s = socket.socket()

read_in_buffer = b''

history = []

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
			history += [new_frame]

			# turn datetime into just clock time for printing
			printable = new_frame[:]
			printable[1] = datetime.strftime(printable[1], '%H:%M:%S.%f')[:-4]
			print('\t'.join(printable))

			read_in_buffer = b''

def parse_frame(frame):
	# ignore the initial 'artifact' field since it is always empty
	elements = frame[1:].split(',')

	# parse time
	timestring = elements[1]
	elements[1] = datetime.strptime(timestring, '%Y/%m/%d %H:%M:%S.%f')
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
