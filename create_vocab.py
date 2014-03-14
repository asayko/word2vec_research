#!/usr/bin/env python

import sys
import words_reader
import multiprocessing
import collections 
import codecs

def calc_sub_dict(fin_name, queue, part, total_parts):
	counter = collections.Counter()
	for word in words_reader.read_words(fin_name, part, total_parts):
		counter[word] += 1
	queue.put(counter)

def calc_dict(fin_name, fout_name, thrds = 1, count_threshhold = 5):
 	
	queue = multiprocessing.Queue()

	ps = []
	for i in range(thrds):
		p = multiprocessing.Process(target=calc_sub_dict, args=(fin_name, queue, i, thrds))
		p.start()
		ps.append(p)

	counter = collections.Counter()
	for p in ps:
		c = queue.get()
		counter.update(c)

	for p in ps:
		p.join()

	total_words = 0
	with codecs.open(fout_name, "w") as fout:
		for word, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
			total_words += count
			if count >= count_threshhold:
				print >> fout, "%s\t%d" % (word.encode("utf-8"), count)

	return total_words


if __name__ == '__main__':
	fin_name = sys.argv[1]
	fout_name = sys.argv[2]
	thrds = 1
	count_threshhold = 5
	if len(sys.argv) >= 4:
		thrds = int(sys.argv[3])
	if len(sys.argv) >= 5:
		count_threshhold = int(sys.argv[4])

	calc_dict(fin_name, fout_name, thrds, count_threshhold)