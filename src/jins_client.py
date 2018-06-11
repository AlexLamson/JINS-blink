# built to connect with and read data from JINS MEME Data Logger

import sys
import socket

s = socket.socket()

read_in_buffer = b''

history = []

def initialize_connection(host, port):
	s.connect( (host, port) )

def read_forever():
	global read_in_buffer, history

	print('Looking for input...')

	while True:
		new_char = s.recv(1)
		read_in_buffer += new_char

		# if there was a newline, write the frame
		if read_in_buffer[-2:] == b'\r\n':
			new_frame_raw = read_in_buffer[:-2].decode('utf-8')
			new_frame = parse_frame(new_frame_raw)
			history += [new_frame]
			print('\t'.join(new_frame))

			read_in_buffer = b''

def parse_frame(frame):
	# ignore the initial 'artifact' field since it is always empty
	elements = frame[1:].split(',')

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
