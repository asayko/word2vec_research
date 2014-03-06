#!/usr/bin/env python

import numpy as np
import heapq
from scipy.spatial.distance import cosine
import sys

class TDictRes:
	def __init__(self, word, dist):
		self.word = word
		self.dist = dist
	def __lt__(self, other):
		return self.dist > other.dist
	def __repr__(self):
		return "%s\t%lf" % (self.word, self.dist)
	def __str__(self):
		return self.__repr__()

class TDemoItem:
	def __init__(self, type, words, n_nearest = 10):
		self.type = type
		self.words = words
		self.status = 'created'
		self.results = []
		self.vec = None
		self.n_nearest = n_nearest

	def try_add_to_res(self, word, vec):
		to_add = TDictRes(word, cosine(self.vec, vec))
		if len(self.results) < self.n_nearest:
			self.results.append(to_add)
		else:
			heapq.heappushpop(self.results, to_add)

	def update(self, word, vec, w2v, n_nearest = 10):
		if self.type == 'nearest_words':
			if self.status == 'out_of_dict':
				return

			if self.vec is None:
				if self.words in w2v:
					self.vec = w2v[self.words]
					self.status = 'found_in_vocab'
				else:
					self.status = 'out_of_dict'
					return

			self.try_add_to_res(word, vec)

		elif self.type == 'nearest_to_logic_operation1_results':
			if self.status == 'out_of_dict':
				return

			if self.words[0] in w2v and self.words[1] in w2v and self.words[2] in w2v:
				self.vec = w2v[self.words[0]] - w2v[self.words[1]] + w2v[self.words[2]]
				self.status = 'found_in_vocab'
			else:
				self.status = 'out_of_dict'
				return

			self.try_add_to_res(word, vec)
		else:
			raise "Unknown demo item type %s." % self.type

	def __repr__(self):
		nearest_words_list = "\n".join([str(x) for x in sorted(self.results, key=lambda y: y.dist)])
		res = ""
		if self.type == 'nearest_words':
			return "For word %s the nearest words are:\n%s\n=========" % (self.words, nearest_words_list)
		elif self.type == 'nearest_to_logic_operation1_results':
			return "For %s - %s + %s the nearest words are:\n%s\n=========" % (self.words[0], self.words[1], self.words[2], nearest_words_list)

		raise "No way to represent demo item of type %s." % self.type

def main():
	demo_items = [TDemoItem('nearest_words', 'putin'),\
	 TDemoItem('nearest_words', 'usa'),\
	 TDemoItem('nearest_words', 'russia'),\
	 TDemoItem('nearest_to_logic_operation1_results', ['putin','russia','usa'])\
	 ]

	repr_fin_name = sys.argv[1]
	w2v = {}

	print "Reading the dict from %s." % repr_fin_name
	for line in open(repr_fin_name, 'r'):
		line_parts = line.split()
		word = line_parts[0]
		vec = np.array([float(x) for x in line_parts[1:]])
		w2v[word] = vec

	print "The dict is read: "
	i=0
	for w, vec in w2v.items():
		print w
		i += 1
		if i == 5:
			break
	print "..."

	print "Processing demo examples."
	for word, vec in w2v.items():
		for demo_item in demo_items:
			demo_item.update(word, vec, w2v)

	for demo_item in demo_items:
		print demo_item

if __name__ == '__main__':
	main()