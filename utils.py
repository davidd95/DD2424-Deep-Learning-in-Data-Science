import numpy as np
import os as os

def get_data():
	""" This function reads in the text in a specified file located in
		the /data folder.

	Args:
		file_name (string): The name of the file to be loaded.

	Returns:
		book_data (string): The training text.
		book_chars (arr(string)): Array that holds the unique chars that is present in training data.
		char_to_ind (dict of string: int): Dict that converts a char to the representing index in the book_chars array.
		int_to_char (dict of int: string): Dict that converts a index to the representing char in the book_chars array.
		K (int): How many unique chars there is in the training data.
	"""
	# Read traning data
	with open(os.path.join('data', 'goblet.txt'),'rb') as book:
		book_data = book.read()

	# Get unique chars
	book_data = "".join(map(chr, book_data))
	book_chars = np.array(list(set(book_data)))
	K = len(book_chars)

	# Create mappings
	char_to_ind = {}
	ind_to_char = {}
	for i in range(K):
		char_to_ind[book_chars[i]] = i
		ind_to_char[i] = book_chars[i]

	return book_data, book_chars, char_to_ind, ind_to_char, K

def relative_error(one, two):
	""" Calculates the relative error between two numbers"""
	two_abs = two
	one_abs = one
	if(one < 0):
		one_abs *= -1
	if(two < 0):
		two_abs *= -1
	return ((np.abs(one-two))/(max(0.0000001, one_abs+two_abs)))