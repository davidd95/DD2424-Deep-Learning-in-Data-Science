import numpy as np
from classes import RNN
from utils import get_data, relative_error

def compare_gradients(book_chars, char_to_ind, ind_to_char, K):
	""" This function compares the analytically derived gradients to numerical estimates """
	model = RNN.RNN(book_chars, char_to_ind, ind_to_char, K, 5)

	grad_w_n, grad_b_n = model.get_num_grads_slow(book_chars[25:50], book_chars[26:51], 1e-4)
	grad_w_a, grad_b_a, _, _ = model.get_gradients(book_chars[25:50], book_chars[26:51])

	maxi = -9999

	error = np.zeros(len(model.biases.keys()))
	for li, layer in enumerate(model.biases.keys()):
		print(layer, '-', model.biases[layer].size)
		for i in range(len(grad_b_a[layer])):
			err = relative_error(grad_b_a[layer][i][0], grad_b_n[layer][i])
			if(err > maxi):
				maxi = err
			if(err > 1e-06):
				error[li] += 1

	print(error)

	error = np.zeros(len(model.weights.keys()))
	for li, layer in enumerate(model.weights.keys()):
		print(layer, '-', model.weights[layer].size)
		for i in range(len(grad_w_a[layer])):
			for j in range(len(grad_w_a[layer][0])):
				err = relative_error(grad_w_a[layer][i][j], grad_w_n[layer][i][j])
				if(err > maxi):
					maxi = err
				if(err > 1e-06):
					error[li] += 1

	print(error)
	print("max", maxi)

def main(check_gradients = False):
	book_data, book_chars, char_to_ind, ind_to_char, K = get_data()
	rnn = RNN.RNN(book_chars, char_to_ind, ind_to_char, K)

	if (check_gradients == True):
		compare_gradients(book_chars, char_to_ind, ind_to_char, K)
	else:
		rnn.fit(book_data, eta = 0.05)

main()