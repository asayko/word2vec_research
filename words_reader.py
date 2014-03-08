#!/usr/bin/env python

import sys
import os
import re
import codecs

def read_words(fin_name, part = 0, total_parts=-1):
	file_size = os.stat(fin_name).st_size

	if total_parts == -1:
		size = file_size
		offset = 0
	else:
		size = file_size / total_parts
		offset = part * size

	fin = codecs.open(fin_name, 'r', 'utf-8', errors='ignore')

	fin.seek(offset)

	i = 0
	while fin.tell() < offset + size:
		line = fin.readline()
		for match in re.finditer("\w+", line, re.UNICODE):
			i += 1
			word = match.group(0)
			yield word

if __name__ == '__main__':
	fin_name = sys.argv[1]
	for word in read_words(fin_name):
		print word.encode("utf-8")