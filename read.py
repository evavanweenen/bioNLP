import numpy as np

import string
import re

from tqdm import tqdm

import nltk
from nltk import word_tokenize
from nltk import pos_tag as POSTag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lem = WordNetLemmatizer()

def postag_to_wordnet(tag):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return None

def preprocess_text(text):
	# TODO: typos and misspellings
	# TODO: identify medical abbreviations
	# TODO: identify diseases and other medical terms
	# not TODO: stemming: text = [PorterStemmer().stem(word) for word in text]

	# remove end of line
	text = text.rstrip('\n')

	# lowercasing
	text = text.lower()

	# remove anything starting with http or www
	text = re.sub(r"http\S+", '', text)
	text = re.sub(r"www.\S+", '', text)

	# punctuation removal (and replace with whitespace)
	text = re.sub(r"""[%s]+\ *"""%string.punctuation, " ", text)

	# tokenization
	text = word_tokenize(text)

	# stop-words removing
	text = [word for word in text if word not in stopwords.words('english')]
	
	# medical abbreviations, phrases, etc.

	# pos tagging
	tag = list(map(lambda x: (x[0], x[1], postag_to_wordnet(x[1])), POSTag(text)))

	# lemmatization (+ replace numbers with a num tag)
	text = ['NUM' if pos == 'CD' else (word if wn is None else lem.lemmatize(word, wn)) for word, pos, wn in tag]

	return text

def preprocess(path_in, path_out):
	if path_in.endswith('val.txt'):
		path_in = path_in[:-7] + 'dev.txt'

	Y = [] # labels

	with open(path_in, 'r') as f_in, open(path_out+'_x.txt', 'w') as f_out:
		for line in tqdm(f_in):
			# remove abstract splits and file IDs as well as clinical trial registry line
			if line.startswith('#') or line == '\n' or 'clinicaltrials.gov' in line.lower():
				continue

			# split by label and text
			label, text = line.split('\t')[:2]

			# preprocess text
			text = preprocess_text(text)

			# only write non-empty lines
			if ' '.join(text).strip() != '':
				# write preprocessed text to file
				f_out.write(' '.join(text) + '\n')
				
				Y.append(label)

	np.savetxt(path_out+'_y.txt', Y, fmt='%s')