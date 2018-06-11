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

		if read_in_buffer[-2:] == b'\r\n':
			newest_frame = read_in_buffer[:-2].decode('utf-8')
			history += [newest_frame]
			print(newest_frame)

			read_in_buffer = b''

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
