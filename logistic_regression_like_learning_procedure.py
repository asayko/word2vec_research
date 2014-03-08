#!/usr/bin/env python

import numpy as np
import words_reader
import create_vocab
import codecs
import sys

class TContext:

	def __init__(self, window_size):
		self.window_size = window_size
		self.words = [None] * window_size
		self.window_begins = 0
		self.window_ends = 0
		self.cur_size = 0
		self.words_processed = 0

	def append_word(self, word):
		if self.cur_size < self.window_size:
			self.words[self.window_ends] = word
			self.window_ends = (self.window_ends + 1) % self.window_size
			self.cur_size += 1
		else:
			self.words[self.window_ends] = word
			self.window_ends = (self.window_ends + 1) % self.window_size
			self.window_begins = (self.window_begins + 1) % self.window_size

	def update_dict(self, pred_dict, context_dict, t1, table_for_random_words_generation, learning_rate, ns = 7):
		i = self.window_begins
		center_word = (self.window_begins + self.cur_size / 2) % self.window_size

		if not self.words[center_word] in pred_dict:
			return

		# learning pred and context dicts
		pred = pred_dict[self. words[center_word]]
		while True:
			if i != center_word:
				if self.words[i] in context_dict:
					t1.fill(0.0)

					context = context_dict[self.words[i]]
					prod = np.dot(context, pred)
					
					s = 1.0 / (1 + np.exp(-prod))
					t = np.exp(prod)
					s_dot = t / ((1 + t) * (1 + t))
					
					t1 = learning_rate * s_dot * (s - 1) * pred
					pred -= learning_rate * s_dot * (s - 1) * context
					context -= t1

					# negative examples sampling
					for tt in xrange(ns):
						rand_word = table_for_random_words_generation[np.random.randint(len(table_for_random_words_generation))]
						rand_pred = pred_dict[rand_word]
						prod = np.dot(context, rand_pred)
						s = 1.0 / (1 + np.exp(-prod))
						t = np.exp(prod)
						s_dot = t / ((1 + t) * (1 + t))

						t1 = learning_rate * s_dot * s * rand_pred
						rand_pred -= learning_rate * s_dot * s * context
						context -= t1

			i = (i + 1) % self.window_size
			if i == self.window_ends:
				break

	def __repr__(self):
		center_word = (self.window_begins + self.cur_size / 2) % self.window_size
		i = self.window_begins
		res = []
		while True:
			if i != center_word:
				res.append(self.words[i])
			else:
				res.append("===>%s<===" % self.words[i])
			i = (i + 1) % self.window_size
			if i == self.window_ends:
				break
		return "\t".join(res)

	def __str__(self):
		return self.__repr__()

def main():
	text_fin_name = sys.argv[1]
	dims = int(sys.argv[2])
	repr_fout_name = sys.argv[3]
	vocab_name = "%s.vocab" % text_fin_name

	print >> sys.stderr, "Processing text to create vocab."
	total_words = create_vocab.calc_dict(text_fin_name, vocab_name, thrds = 10, count_threshhold = 4)

	init_learning_rate = 0.03
	learning_rate = init_learning_rate
	delta_learning_rate = init_learning_rate / total_words

	table_for_random_words_generation = []

	print >> sys.stderr, "Initializing in memory dicts."
	pred_dict = {}
	context_dict = {}
	for line in codecs.open(vocab_name, 'r', 'utf-8'):
		line_parts = line.split()
		word = line_parts[0]
		freq = int(line_parts[1])
		pred_dict[word] = np.zeros(dims)
		context_dict[word] = np.random.uniform(-0.5, 0.5, dims)
		if len(table_for_random_words_generation) < 2e8:
			table_for_random_words_generation.extend([word] * int(pow(freq, 3.0 / 4)))
	
	print >> sys.stderr, "Processing text and learning."
	words_read = 0
	context = TContext(5)
	t1 = np.ones(dims)
	for word in words_reader.read_words(text_fin_name):
		words_read += 1
		learning_rate = max(learning_rate, 1e-9)
		context.append_word(word)
		context.update_dict(pred_dict, context_dict, t1, table_for_random_words_generation, learning_rate, 5)
		learning_rate -= delta_learning_rate
		if words_read % 100000 == 0:
			print >> sys.stderr, "%d words are read." % (words_read)

	print >> sys.stderr, "Saving new features."
	with open(repr_fout_name, "w") as fout:
		for word, repres in context_dict.items():
			print >> fout, "%s\t%s" % (word.encode('utf-8'), "\t".join(["%lf" % i for i in repres.tolist()]))

if __name__ == '__main__':
	main()