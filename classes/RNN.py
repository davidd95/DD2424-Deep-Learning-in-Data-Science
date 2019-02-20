import numpy as np
import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

class RNN():
	"""This is a class for a simple RNN - for more information see the pdf in the parent
	folder.

	Attributes:
		eps (float64): Small number to avoid numerical over/under-flows.
		chars (arr(string)): Array that holds the unique chars that is present in training data.
		char_to_ind (dict of string: int): Dict that converts a char to the representing index in the chars array.
		int_to_char (dict of int: string): Dict that converts a index to the representing char in the chars array.
		K (int): How many unique chars there is in chars.
		m (int): Sets the dimension of the hidden layer.
		biases (dict of string: arr(float64)): The biases for this NN.
		weights (dict of string: arr(float64)): The weights for this NN.
	"""
	def __init__(self, chars = None, char_to_ind = {}, ind_to_char = {}, K = 26, m = 100):
		"""Function is called upon initialization of the class. Sets up the weights and biases."""

		super(RNN, self).__init__()

		np.random.seed(7)
		self.eps = 1e-3

		self.chars = chars
		self.char_to_ind = char_to_ind
		self.ind_to_char = ind_to_char

		self.K = K
		self.m = m

		# Initialize biases as zero vectors
		self.biases = {
			'b' : np.asfortranarray(np.zeros((K, 1), dtype="float64")),
			'c'	: np.asfortranarray(np.zeros((m, 1), dtype="float64"))
		}

		# Initialize weights as normally distributed random numbers
		sig = .01
		self.weights = {
			'U' : np.asfortranarray(np.random.randn(m, K).astype(dtype="float64") * sig),
			'W' : np.asfortranarray(np.random.randn(m, m).astype(dtype="float64") * sig),
			'V' : np.asfortranarray(np.random.randn(K, m).astype(dtype="float64") * sig)
		}

	def __call__(self, h, x, n):
		"""Makes the object callable.

		This function will synthesize("sample") text from the NN.

		Args:
			h (arr(float64)): The initial hidden state.
			x (arr(float64)): Start character, one-hot encoded.
			n (int): How many characters to sample.

		Returns:
			A string that has been synthesized from the NN.

		"""
		h_t = h
		x_t = x

		seq = np.zeros(n, dtype='str')

		for i in range(n):
			x_next, x_next_oh, h_next, _ = self.forward(h_t, x_t)
			seq[i], h_t, x_t = x_next, h_next, x_next_oh

		return ''.join(seq)

	def forward(self, h_t, x_t):
		"""Make one forward-pass.

		Note:
			This is not the whole process that loops over tau. This is the
			procedure for one isolated char.

		Args:
			h_t (arr(float64)): The hidden state.
			x_t (arr(float64)): Current char, one-hot encoded.

		Returns:
			x_next (string): The next sampled char.
			x_t (arr(float64)): The next sampled char, one-hot encoded.
			h_t (arr(float64)): The next hidden state.
			p_t (arr(float64)): The probability distribution over possible next chars.

		"""
		h_t, x_t = self.ensure_fortran([h_t, x_t])

		a_t = np.matmul(self.weights['W'], h_t) + np.matmul(self.weights['U'], x_t) + self.biases['c']
		h_t = np.tanh(a_t)
		o_t = np.matmul(self.weights['V'], h_t) + self.biases['b']
		p_t = self.softmax(o_t)

		x_next = np.random.choice(self.chars, p = p_t.flatten())
		x_t = np.zeros((self.K, 1))
		x_t[self.char_to_ind[x_next]] = 1

		return x_next, x_t, h_t, p_t

	def backward(self, x_t, h_t, p_t, y_t, h_prev, dh_next):
		"""Make one backward-pass.

		Note:
			This is not the whole process that loops over tau. This is the
			procedure for one isolated char.

		Args:
			x_t (arr(float64)): Current char, one-hot encoded.
			h_t (arr(float64)): The hidden state.
			p_t (arr(float64)): The probability distribution over possible next chars.
			y_t (arr(float64)): Ground trouth of next chars from training_data.
			h_prev (arr(float64)): The previous hidden state.

		Returns:
			dV (arr(float64)): Gradients of V.
			dW (arr(float64)): Gradients of W.
			dU (arr(float64)): Gradients of U.
			db (arr(float64)): Gradients of b.
			dc (arr(float64)): Gradients of c.
		"""
		x_t, h_t, p_t, y_t, h_prev, dh_next = self.ensure_fortran([x_t, h_t, p_t, y_t, h_prev, dh_next])

		# Calc general deltas
		g_t = -np.subtract(y_t, p_t)
		dh = np.matmul(self.weights['V'].T, g_t) + dh_next
		da = np.multiply((1. - np.multiply(h_t, h_t)), dh)

		# Calc dL/dV
		dV = np.matmul(g_t, h_t.T)

		# Calc dL/dW
		dW = np.matmul(da, h_prev.T)

		# Calc dL/dU
		dU = np.matmul(da, x_t.T)

		# Biases
		db = g_t
		dc = da

		dh_next = np.matmul(self.weights['W'].T, da)

		return dV, dW, dU, db, dc, dh_next

	def get_gradients(self, seq, labels, h_t = np.array([])):
		"""Get the gradients given a specific sequence.

		Args:
			seq (string): The sequence.
			labels (string): The string of "next-chars" aka. labels.
			h_t (arr(float64)): The initial hidden-state.

		Returns:
			d_weights (dict of string: arr(float64)): Gradients of the weights.
			d_biases (dict of string: arr(float64)): Gradients of the biases.
			h_t (arr(float64)): The last hidden-state.
			loss (float64): The loss for this sequence.
		"""
		# set up dicts
		d_biases = {
			'b' : np.asfortranarray(np.zeros((self.K, 1), dtype="float64")),
			'c'	: np.asfortranarray(np.zeros((self.m, 1), dtype="float64"))
		}
		d_weights = {
			'U' : np.asfortranarray(np.zeros((self.m, self.K), dtype="float64")),
			'W' : np.asfortranarray(np.zeros((self.m, self.m), dtype="float64")),
			'V' : np.asfortranarray(np.zeros((self.K, self.m), dtype="float64"))
		}
		h = {}
		p = {}

		# forward
		if (h_t.size == 0):
			h_t = np.zeros((self.m, 1))
		h[-1] = copy.deepcopy(h_t)
		loss = 0
		for t, x in enumerate(seq):
			x_t = np.zeros((self.K, 1))
			x_t[self.char_to_ind[x]] = 1

			y_t = np.zeros((self.K, 1))
			y_t[self.char_to_ind[labels[t]]] = 1

			_, _, h_t, p_t = self.forward(h_t, x_t)

			h[t] = copy.deepcopy(h_t).astype(dtype="float64")
			p[t] = copy.deepcopy(p_t).astype(dtype="float64")

			loss += np.log(np.matmul(y_t.T, p_t))

		# backward
		dh_next = np.zeros(h[0].shape)
		for t in range(len(seq)-1, -1, -1):
			x_t = np.zeros((self.K, 1))
			x_t[self.char_to_ind[seq[t]]] = 1

			y_t = np.zeros((self.K, 1))
			y_t[self.char_to_ind[labels[t]]] = 1

			dV, dW, dU, db, dc, dh_next = self.backward(x_t, h[t], p[t], y_t, h[t-1], dh_next)

			d_biases = {
				'b' : d_biases['b'] + db,
				'c'	: d_biases['c'] + dc
			}
			d_weights = {
				'U' : d_weights['U'] + dU,
				'W' : d_weights['W'] + dW,
				'V' : d_weights['V'] + dV
			}

		# Clip to avoid exp/van gradients
		d_biases = {
			'b' : np.maximum(np.minimum(d_biases['b'], 5), -5),
			'c'	: np.maximum(np.minimum(d_biases['c'], 5), -5)
		}
		d_weights = {
			'U' : np.maximum(np.minimum(d_weights['U'], 5), -5),
			'W' : np.maximum(np.minimum(d_weights['W'], 5), -5),
			'V' : np.maximum(np.minimum(d_weights['V'], 5), -5)
		}

		return d_weights, d_biases, h_t, -loss[0][0]

	def compute_loss(self, seq, labels):
		"""Get the loss given a specific sequence.

		Args:
			seq (string): The sequence.
			labels (string): The string of "next-chars" aka. labels.

		Returns:
			The loss for this sequence.
		"""
		h_t = np.zeros((self.m, 1))
		loss = 0
		for i, x in enumerate(seq):
			x_t = np.zeros((self.K, 1))
			x_t[self.char_to_ind[x]] = 1
			_, _, h_t, p_t = self.forward(h_t, x_t)

			y_t = np.zeros((self.K, 1))
			y_t[self.char_to_ind[labels[i]]] = 1

			loss += np.log(np.matmul(y_t.T, p_t))
		return -loss[0][0]

	def fit(self, data = None, eta = .1, seq_length = 25):
		"""The function that trains the model.

		It will print its progress every 1000th iteration. Alongside this it will save
		a graph with the evolution of the loss in the folder /graphs.

		Args:
			data (string): The training text.
			eta (float64): The learning rate of which the params will be updated.
			seq_length (int): How long should the individual sub-sequences be?
		"""
		m = {
			'b' : np.zeros(self.biases['b'].shape),
			'c' : np.zeros(self.biases['c'].shape),
			'W' : np.zeros(self.weights['W'].shape),
			'U' : np.zeros(self.weights['U'].shape),
			'V' : np.zeros(self.weights['V'].shape)
		}

		smooth_loss = None
		num_iter = 0
		max_iters = 160000
		epoch = 1
		h_t = np.zeros((self.m, 1))
		e = 0
		all_losses = np.zeros(max_iters)
		print(f'Epoch #{epoch}')
		while (num_iter < max_iters):
			if ((e+seq_length+1) > len(data)):
				# We have gone trough the training data once..
				self.document_iter(e, num_iter, smooth_loss, h_t, all_losses)

				e = 0
				epoch += 1
				h_t = np.zeros((self.m, 1))
				print(f'Epoch #{epoch}')

				continue

			seq = data[e:e+seq_length]
			labels = data[e+1:e+seq_length+1]

			d_weights, d_biases, h_t, loss = self.get_gradients(seq, labels, h_t)

			# Update weights
			for key in d_weights.keys():
				to_be_updated = self.weights[key]
				gradients = d_weights[key]
				self.weights[key], m[key] = self.ada_grad(eta, m[key], to_be_updated, gradients)

			# Update biases
			for key in d_biases.keys():
				to_be_updated = self.biases[key]
				gradients = d_biases[key]
				self.biases[key], m[key] = self.ada_grad(eta, m[key], to_be_updated, gradients)

			# Update the smooth loss, if None use the current loss
			if (smooth_loss == None):
				smooth_loss = loss
			else:
				smooth_loss = .999 * smooth_loss + .001 * loss

			all_losses[num_iter] = smooth_loss

			if (num_iter % 1000 == 0):
				self.document_iter(e, num_iter, smooth_loss, h_t, all_losses)

			e += seq_length
			num_iter += 1

		print(f'\nDone with {num_iter} of iterations. (Epoch {epoch}).')
		print(f'Min. loss was {all_losses.min()}, and that was achived at iter #{np.argmin(all_losses)}.')

	def ada_grad(self, eta, m, theta, grad):
		"""This function performs the AdaGrad calculations.

		Args:
			eta (float64): The learning rate of which the params will be updated.
			m (arr(float64)): The previous m.

		Returns:
			updated (arr(float64)): The updated weight/biases.
			m (arr(float64)): The new m.
		"""
		m += np.multiply(grad, grad)
		updated = theta - np.multiply((eta/(np.sqrt(m + self.eps))), grad)

		return updated, m

	""" Beyond this point lays utility land :) """
	def softmax(self, s):
		"""Calculated softmax of given matrix/float s.

		Args:
			s (float64)/(arr(float64)): The value/s to be softmaxed.

		Returns:
			The softmaxed values.
		"""
		exp = np.exp(s)
		return exp/np.sum(exp)

	def ensure_fortran(self, args = []):
		"""Check if array is stored the fortran way, if not convert it.

		Args:
			args (list of any(any)): The arrays that should be checked.

		Returns:
			Only fortran stored arrays.
		"""
		for i, arg in enumerate(args):
			if(not arg.flags.f_contiguous):
				args[i] = np.asfortranarray(arg)

		return args

	def document_iter(self, e, num_iter, smooth_loss, h_t, all_losses):
		"""Print process to stdout and save graph."""
		print(f'Processing char #{e} in the {num_iter} iter')
		print(f'	Smooth loss: {smooth_loss}')
		x_start = np.zeros((self.K, 1))
		x_start[self.char_to_ind[' ']] = 1
		print(f'	Test text:\n{self(h_t, x_start, 200)}\n')
		plt.plot(np.arange(num_iter), all_losses[0:num_iter])
		plt.savefig(os.path.join('graphs', 'smooth_loss_' + str(num_iter)))
		plt.close()

	def get_num_grads_slow(self, X, Y, h = 1e-5):
		"""Function that estimates the gradients numerically.

		Note:
			See: https://en.wikipedia.org/wiki/Numerical_differentiation

		Args:
			X (string): The current sequence.
			Y (string): The string of "next-chars" aka. labels.
			h (float64): Very small number used when estimating.

		Returns:
			grad_W (dict of string: arr(float64)): Estimated gradients of the weights.
			grad_b (dict of string: arr(float64)): Estimated gradients of the biases.
		"""
		grad_b = {}
		grad_W = {}

		for layer_ind, layer in enumerate(self.biases.keys()):
			grad_b[layer] = np.zeros(len(self.biases[layer]))
			for i in range(len(self.biases[layer])):
				b_try = np.array(self.biases[layer], copy = True, dtype="float64")
				old_b = np.array(self.biases[layer], copy = True, dtype="float64")
				b_try[i] = b_try[i] - h
				self.biases[layer] = b_try
				c1 = self.compute_loss(X, Y)
				self.biases[layer] = old_b

				b_try = np.array(self.biases[layer], copy = True, dtype="float64")
				old_b = np.array(self.biases[layer], copy = True, dtype="float64")
				b_try[i] = b_try[i] + h
				self.biases[layer] = b_try
				c2 = self.compute_loss(X, Y)
				self.biases[layer] = old_b

				grad_b[layer][i] = (c2-c1) / (2.*h)

		for layer_ind, layer in enumerate(self.weights.keys()):
			grad_W[layer] = np.zeros(self.weights[layer].shape)
			for i in range(len(self.weights[layer])):
				for j in range(len(self.weights[layer][0])):
					W_try = copy.deepcopy(np.array(self.weights[layer], copy = True, dtype="float64"))
					W_old = copy.deepcopy(np.array(self.weights[layer], copy = True, dtype="float64"))
					W_try[i][j] = W_try[i][j] - h
					self.weights[layer] = W_try
					c1 = self.compute_loss(X, Y)
					self.weights[layer] = W_old

					W_try = copy.deepcopy(np.array(self.weights[layer], copy = True, dtype="float64"))
					W_old = copy.deepcopy(np.array(self.weights[layer], copy = True, dtype="float64"))
					W_try[i][j] = W_try[i][j] + h
					self.weights[layer] = W_try
					c2 = self.compute_loss(X, Y)
					self.weights[layer] = W_old
					grad_W[layer][i][j] = (c2-c1) / (2.*h)

		return grad_W, grad_b