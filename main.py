import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"

import numpy as np
import pandas as pd

from read import preprocess

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.manifold import TSNE

import multiprocessing
import pickle

from gensim.models import Word2Vec

from umap import UMAP

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm
import gc

sns.set()
sns.set_context('paper')
sns.set_style('white')

# ------------------------ data
# preprocess
for split in ('train', 'val', 'test'):
	# read and preprocess
	preprocess(path_in='./data/PubMed_200k_RCT/'+split+'.txt', path_out='./data/'+split)

# open
def read_text(datadir, split):
	with open(datadir+split+'_x.txt', 'r') as file:
		text = file.read().split('\n')[:-1]
	return text

data = {'x'	: {split: read_text('./data/', split) for split in ('train', 'val', 'test')},
		'y'	: {split: np.loadtxt('./data/'+split+'_y.txt', dtype='str') for split in ('train', 'val', 'test')}}

# ------------------------ plot
# label distribution
pd.Series(data['y']['train']).value_counts().plot.bar()
plt.show() 

# ------------------------ tf idf
# binarize labels
le = LabelEncoder()
le.classes_ = np.array(['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'])
Y = {split : le.transform(data['y'][split]) for split in ('train', 'val', 'test')}

# transform text with tf idf
# min_df: we keep only words that appear in 100 or more documents
# ngram_range: try ngrams of size 1 and 2
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=100, ngram_range=(1,2), norm='l2')

X = {'train': tfidf.fit_transform(data['x']['train']),
	 'val'	: tfidf.transform(data['x']['val']),
	 'test'	: tfidf.transform(data['x']['test'])}

words = np.array(tfidf.get_feature_names())
print("Number of unique Unigrams and bigrams: ", len(words))

# print features with highest chi2
print("Features with highest chi2: ")
for l, lname in enumerate(le.classes_):
	chi2_words = chi2(X['train'], Y['train'] == l)
	words_sorted = words[np.argsort(chi2_words[0])[::-1]]
	print(lname, words_sorted[:5])

# regression
clf = LogisticRegression(penalty='l1', solver='liblinear')
clf.fit(X['train'], Y['train'])

# write to file
with open('./model/tfidf-classifier.pkl', 'wb') as file:
	pickle.dump(clf, file)

# read model
with open('./model/tfidf-classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

# plot features with highest coefficient
coefs = pd.DataFrame(clf.coef_.T, columns=le.classes_, index=words)
for i, lab in enumerate(le.classes_):
	idx = coefs[lab].abs().argsort()
	coefs[lab][idx][-20:].plot(kind='barh')
	plt.title(lab)
	plt.show()

# classification report
print("\n -------------- train -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['train'], clf.predict(X['train']), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- val -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['val'], clf.predict(X['val']), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- test -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['test'], clf.predict(X['test']), 
	target_names=le.classes_, output_dict=True)).T)

# final score
print("F1-score (weighted) for test set: ", f1_score(Y['test'], clf.predict(X['test']), average='weighted'))

# confusion matrix
plot_confusion_matrix(clf, X['test'], Y['test'], display_labels=le.classes_, 
	normalize='true', xticks_rotation='vertical', cmap='Blues')
plt.xlabel('Predicted label') ; plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# roc curve
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(le.classes_)):
	fpr[i], tpr[i], _ = roc_curve(Y['test'] == i, clf.decision_function(X['test'])[:,i])
	roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(le.classes_)):
	plt.plot(fpr[i], tpr[i], label=le.classes_[i]+r' (AUC = %.2f)'%roc_auc[i])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

del X ; gc.collect()

# ------------------------ word2vec
# settings
WINDOW_SIZE = 5 # size of window of word2vec
VECTOR_SIZE = 200 # size of the embedding layer in word2vec
EPOCHS = 30 # number of epochs to train for

cores = multiprocessing.cpu_count()

# create corpus
X_corpus = {split: [line.split() for line in data['x'][split]] for split in ('train', 'val', 'test')}

# build and train word2vec model
w2v = Word2Vec(min_count=100, window=WINDOW_SIZE, vector_size=VECTOR_SIZE, sg=1, workers=cores-1)
w2v.build_vocab(X_corpus['train'], progress_per=10000)
w2v.train(X_corpus['train'], total_examples=w2v.corpus_count, epochs=EPOCHS, report_delay=1)

w2v.save('./model/word2vec.model')

# load model
w2v = Word2Vec.load('./model/word2vec.model')

# transform lines to their vector embeddings
X = {split : np.full([len(data['x'][split]), 200], np.nan) for split in ('train', 'val', 'test')}
for split in ('train', 'val', 'test'):
	for i, line in enumerate(data['x'][split]):
		emb = []
		for word in line.split():
			try:
				emb.append(w2v.wv[word])
			except KeyError:
				emb.append(np.full([200], np.nan)) # replace with nan if word unknown
		X[split][i] = np.nanmean(emb, axis=0) # ignore nans

	# remove nan rows (for this it would have been better to use fasttext)
	mask = ~np.isnan(np.sum(X[split], axis=1))
	Y[split] = Y[split][mask]
	X[split] = X[split][mask]

	# save so we can skip this step later
	np.save('./data/word2vec/'+split+'_xemb.npy', X[split])
	np.save('./data/word2vec/'+split+'_yemb.npy', Y[split]) # save y because we removed nan rows

# load embeddings
X = {split : np.load('./data/word2vec/'+split+'_xemb.npy') for split in ('train', 'val', 'test')}
Y = {split : np.load('./data/word2vec/'+split+'_yemb.npy') for split in ('train', 'val', 'test')}

# t-sne plot for the embeddings
X_umap = UMAP().fit_transform(X['train'])
np.save('./data/word2vec/umap.npy', X_umap)

umap = np.load('./data/word2vec/umap.npy')
sns.scatterplot(x=umap[0], y=umap[1], hue=Y['train'])
plt.show()

# standardize
scaler = StandardScaler()
X = {'train': scaler.fit_transform(X['train']),
	 'val'	: scaler.transform(X['val']),
	 'test'	: scaler.transform(X['test'])}

# regression
clf = LogisticRegression(penalty='l1', solver='liblinear')
clf.fit(X['train'], Y['train'])

# write to file
with open('./model/word2vec-classifier.pkl', 'wb') as file:
	pickle.dump(clf, file)

# read model
with open('./model/word2vec-classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

# classification report
print("\n -------------- train -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['train'], clf.predict(X['train']), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- val -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['val'], clf.predict(X['val']), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- test -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y['test'], clf.predict(X['test']), 
	target_names=le.classes_, output_dict=True)).T)

# final score
print("F1-score (weighted) for test set: ", f1_score(Y['test'], clf.predict(X['test']), average='weighted'))

# confusion matrix
plot_confusion_matrix(clf, X['test'], Y['test'], display_labels=le.classes_, 
	normalize='true', xticks_rotation='vertical', cmap='Blues')
plt.xlabel('Predicted label') ; plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# roc curve
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(le.classes_)):
	fpr[i], tpr[i], _ = roc_curve(Y['test'] == i, clf.decision_function(X['test'])[:,i])
	roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(le.classes_)):
	plt.plot(fpr[i], tpr[i], label=le.classes_[i]+r' (AUC = %.2f)'%roc_auc[i])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

del X ; gc.collect()


# ------------------------ Bert
# model
class BertClassifier(nn.Module):
	def __init__(self, freeze_bert=False):
		"""
		BlueBert model with simple classifier on top

		freeze_bert : 	Set False to fine-tune the BERT model
		"""
		super(BertClassifier, self).__init__()

		# BERT model
		self.bert = BertModel.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")

		# classifier
		self.classifier = nn.Linear(768, len(le.classes_))

		# freeze the BERT model
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

	def forward(self, X, A):
		"""
		Model in sequential order: BERT - classifier
		"""
		# Feed input to BERT model
		B = self.bert(input_ids=X, attention_mask=A, token_type_ids=None)

		# Feed the last hidden state of the token `[CLS]` to the classifier
		Y_logit = self.classifier(B[0][:,0,:])

		return Y_logit

# settings
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 2e-5
EPSILON = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data
le = LabelEncoder()
le.classes_ = np.array(['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'])
Y = {split : torch.tensor(le.transform(data['y'][split])) for split in ('train', 'val', 'test')}

tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")

X, A = {}, {}
for split in ('train', 'val', 'test'):
	X[split], _, A[split] = tokenizer.batch_encode_plus(data['x'][split], 
		padding=True, truncation=True, max_length=80, return_tensors="pt").values()
	torch.save(X[split], './data/bert/bluebert_x_'+split)
	torch.save(A[split], './data/bert/bluebert_a_'+split)

X = {split : torch.load('./data/bert/bluebert_x_'+split) for split in ('train', 'val', 'test')}
A = {split : torch.load('./data/bert/bluebert_a_'+split) for split in ('train', 'val', 'test')}

dataset = {split: TensorDataset(X[split], A[split], Y[split]) for split in ('train', 'val', 'test')}

dataloader = {'train': DataLoader(dataset['train'], sampler=RandomSampler(dataset['train']), batch_size=BATCH_SIZE),
			  'val'	 : DataLoader(dataset['val'], sampler=SequentialSampler(dataset['val']), batch_size=BATCH_SIZE),
			  'test' : DataLoader(dataset['test'], sampler=SequentialSampler(dataset['test']), batch_size=BATCH_SIZE)}

# model
model = BertClassifier(freeze_bert=False)
model = nn.DataParallel(model) # use multiple GPUs
model.to(device)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

# scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
	num_training_steps=len(dataloader['train'])*EPOCHS)

# total loss over all batches
total_loss = {split: np.zeros(EPOCHS) for split in ('train', 'val', 'test')}

# keep predictions and labels, so that we can calculate the classification report later
# note that Y_true is Y, but then shuffled
Y_pred = {split: {e: [] for e in range(EPOCHS)} for split in ('train', 'val')}
Y_true = {split: {e: [] for e in range(EPOCHS)} for split in ('train', 'val')}

for epoch in range(EPOCHS):

	# training phase
	model.train()

	for i, batch in enumerate(tqdm(dataloader['train'])):
		x, a, y = tuple(b.to(device) for b in batch)

		model.zero_grad()

		# forward pass
		y_pred = model(x, a)

		# compute loss
		loss = loss_fn(y_pred, y)

		# append batch calculations
		Y_pred['train'][epoch].append(y_pred.detach().cpu())
		Y_true['train'][epoch].append(y.detach().cpu())

		total_loss['train'][epoch] += loss.item()

		# backward pass
		loss.backward()

		# clip norm of gradients to one to prevent exploding gradient # TODO: should I do this?
		torch.nn.utils.clip_grad_norm(model.parameters(), 1.)

		# update parameters and learning rate
		optimizer.step()
		scheduler.step()

	# aggregate batch calculations
	Y_pred['train'][epoch] = torch.vstack(Y_pred['train'][epoch])
	Y_true['train'][epoch] = torch.hstack(Y_true['train'][epoch])

	total_loss['train'][epoch] /= len(dataloader['train'])

	# save Y_true and Y_pred
	torch.save(Y_true['train'], './data/bert/bluebert_y_train')
	torch.save(Y_pred['train'], './data/bert/bluebert_ypred_train')

	# save total loss
	torch.save(total_loss, './model/bluebert-loss')

	# save model
	torch.save({'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				'./model/bluebert-classifier-%s.model'%epoch)

	# evaluate model on validation set after each epoch
	model.eval()

	for i, batch in enumerate(tqdm(dataloader['val'])):
		x, a, y = tuple(b.to(device) for b in batch)

		with torch.no_grad():
			# forward pass
			y_pred = model(x, a)

		# compute loss
		loss = loss_fn(y_pred, y)

		# append batch calculations
		Y_pred['val'][epoch].append(y_pred.detach().cpu())
		Y_true['val'][epoch].append(y.detach().cpu())

		total_loss['val'][epoch] += loss.item()

	# aggregate batch calculations
	Y_pred['val'][epoch] = torch.vstack(Y_pred['val'][epoch])
	Y_true['val'][epoch] = torch.hstack(Y_true['val'][epoch])

	total_loss['val'][epoch] /= len(dataloader['val'])

	# save Y_true and Y_pred
	torch.save(Y_true['val'], './data/bert/bluebert_y_val')
	torch.save(Y_pred['val'], './data/bert/bluebert_ypred_val')

	# save total loss
	torch.save(total_loss, './model/bluebert-loss')

# ------------ Evaluate best Bert model
total_loss = torch.load('./model/bluebert-loss')

Y_true = {split: torch.load('./data/bert/bluebert_y_'+split) for split in ('train', 'val')}
Y_pred = {split: torch.load('./data/bert/bluebert_ypred_'+split) for split in ('train', 'val')}

# pick model with lowest validation loss
best_epoch = np.argmin(total_loss['val'])
print("Best epoch: ", best_epoch)

# plot loss over epochs
pd.DataFrame.from_dict(total_loss)[['train', 'val']].plot()
plt.xlabel('Epoch') ; plt.ylabel('CrossEntropy')
plt.show()

# load model
model = BertClassifier(freeze_bert=True)
model = nn.DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

checkpoint = torch.load('./model/bluebert-classifier-%s.model'%best_epoch)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# select only column of Y of best epoch
Y_true = {split: Y_true[split][best_epoch] for split in ('train', 'val')}
Y_pred = {split: Y_pred[split][best_epoch] for split in ('train', 'val')}
Y_true['test'], Y_pred['test'] = [], []

total_loss = {split: total_loss[split][best_epoch] for split in ('train', 'val', 'test')}

# evaluate model on test set
model.eval()

for i, batch in enumerate(tqdm(dataloader['test'])):
	x, a, y = tuple(b.to(device) for b in batch)

	with torch.no_grad():
		# forward pass
		y_pred = model(x, a)

	# compute loss
	loss = loss_fn(y_pred, y)

	# append batch calculations
	Y_pred['test'].append(y_pred.detach().cpu())
	Y_true['test'].append(y.detach().cpu())

	total_loss['test'] += loss.item()

# aggregate batch calculations
Y_pred['test'] = torch.vstack(Y_pred['test'])
Y_true['test'] = torch.hstack(Y_true['test'])

total_loss['test'] /= len(dataloader['test'])

# save Y_true and Y_pred
torch.save(Y_true['test'], './data/bert/bluebert_y_test')
torch.save(Y_pred['test'], './data/bert/bluebert_ypred_test')

# load Y_true and Y_pred
Y_true['test'] = torch.load('./data/bert/bluebert_y_test')
Y_pred['test'] = torch.load('./data/bert/bluebert_ypred_test')

# classification report
print("\n -------------- train -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y_true['train'], torch.argmax(Y_pred['train'], dim=1), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- val -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y_true['val'], torch.argmax(Y_pred['val'], dim=1), 
	target_names=le.classes_, output_dict=True)).T)
print("\n -------------- test -------------- : \n", 
	pd.DataFrame.from_dict(classification_report(Y_true['test'], torch.argmax(Y_pred['test'], dim=1), 
	target_names=le.classes_, output_dict=True)).T)

# final score
print("F1-score (weighted) for test set: ", f1_score(Y_true['test'], torch.argmax(Y_pred['test'], dim=1), average='weighted'))

# confusion matrix
cm = confusion_matrix(Y_true['test'], torch.argmax(Y_pred['test'], dim=1), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_) 
disp.plot(xticks_rotation='vertical', cmap='Blues')
plt.xlabel('Predicted label') ; plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# roc curve
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(le.classes_)):
	fpr[i], tpr[i], _ = roc_curve(Y_true['test'] == i, F.softmax(Y_pred['test'], dim=1)[:,i]) # TODO
	roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(le.classes_)):
	plt.plot(fpr[i], tpr[i], label=le.classes_[i]+r' (AUC = %.2f)'%roc_auc[i])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()