import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation, Convolution1D, GlobalAveragePooling1D, Input, Bidirectional
import random
import sklearn.metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
random.seed(1)
tf.random.set_seed(2)

EPOCHS = 10
embed_size = 50
maxl = 20

class CNN(object):
	"""docstring for CNN"""
	def __init__(self, train_dict):
		self.tok = Tokenizer()
		self.tok.fit_on_texts(train_dict[user_id][post_no] for user_id in train_dict.keys() for post_no in ['0', '1', '2'])
		self.x = self.vecify(np.array([train_dict[user_id]['0'] + ' ' + train_dict[user_id]['1'] + ' ' + train_dict[user_id]['2'] for user_id in train_dict.keys()]))
		self.y = np.array([train_dict[user_id]['label'] for user_id in train_dict.keys()])

		inp = Input(shape=(3*maxl,), dtype = 'int32')
		emb = Embedding(len(self.tok.word_index)+1, embed_size, mask_zero=False)(inp)
		conv = Sequential()
		conv.add(Convolution1D(15, 3, activation='linear', input_shape=(3*maxl, embed_size) ))
		conv.add(Convolution1D(15, 3, activation='linear'))
		conv.add(Activation('relu'))
		conv.add(GlobalAveragePooling1D())
		cnn1 = conv(emb)
		cnn1 = Dense(4, activation='relu')(cnn1)
		cnn2 = Dense(2, activation='softmax')(cnn1)

		self.cnn_model = Model(inputs=inp, outputs=cnn2)

	def vecify(self, texts):
		seqs = pad_sequences(self.tok.texts_to_sequences(texts), maxlen = 3*maxl)
		#exit()
		return seqs

	def train(self):
		self.cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics="accuracy")
		print(self.cnn_model.summary())
		self.cnn_model.fit(self.x, self.y, epochs=EPOCHS, batch_size = 16)

	def test(self, test_dict):
		self.xt = self.vecify(np.array([test_dict[user_id]['0'] + ' ' + test_dict[user_id]['1'] + ' ' + test_dict[user_id]['2'] for user_id in test_dict.keys()]))
		self.yt = np.array([test_dict[user_id]['label'] for user_id in test_dict.keys()])
		print(self.yt)

		out = self.cnn_model.predict(self.xt, batch_size = 16)
#		print(out)
#		exit()
#		self.yp = np.where(out >= 0.5, 1, 0)
		self.yp = np.argmax(out, axis=1)
		self.yt2 = np.argmax(self.yt, axis=1)
		print((out))


		f1 = sklearn.metrics.f1_score(self.yt2, self.yp, pos_label=1, average='binary')
		p = sklearn.metrics.precision_score(self.yt2, self.yp, pos_label=1, average='binary')
		r = sklearn.metrics.recall_score(self.yt2, self.yp, pos_label=1, average='binary')
		print('F1 Precion Recall')
		print(f1, p, r)

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, train_dict):
		self.tok = Tokenizer()
		self.tok.fit_on_texts(train_dict[user_id][post_no] for user_id in train_dict.keys() for post_no in ['0', '1', '2'])
		self.x = self.vecify(np.array([train_dict[user_id]['0'] + ' ' + train_dict[user_id]['1'] + ' ' + train_dict[user_id]['2'] for user_id in train_dict.keys()]))
		self.y = np.array([train_dict[user_id]['label'] for user_id in train_dict.keys()])

		inp = Input(shape=(3*maxl,), dtype = 'int32')
		emb = Embedding(len(self.tok.word_index)+1, embed_size, mask_zero=False)(inp)
		rnn = Sequential()
		rnn.add(Bidirectional(LSTM(12, return_sequences=False), input_shape=(None, embed_size)))
		rnn1 = rnn(emb)
		rnn1 = Dense(4, activation='relu')(rnn1)
		rnn2 = Dense(2, activation='softmax')(rnn1)

		self.rnn_model = Model(inputs=inp, outputs=rnn2)

	def vecify(self, texts):
		seqs = pad_sequences(self.tok.texts_to_sequences(texts), maxlen = 3*maxl)
		#exit()
		return seqs

	def train(self):
		self.rnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics="accuracy")
		print(self.rnn_model.summary())
		self.rnn_model.fit(self.x, self.y, epochs=EPOCHS, batch_size = 16)

	def test(self, test_dict):
		self.xt = self.vecify(np.array([test_dict[user_id]['0'] + ' ' + test_dict[user_id]['1'] + ' ' + test_dict[user_id]['2'] for user_id in test_dict.keys()]))
		self.yt = np.array([test_dict[user_id]['label'] for user_id in test_dict.keys()])
		print(self.yt)

		out = self.rnn_model.predict(self.xt, batch_size = 16)
#		print(out)
#		exit()
#		self.yp = np.where(out >= 0.5, 1, 0)
		self.yp = np.argmax(out, axis=1)
		self.yt2 = np.argmax(self.yt, axis=1)
		print((out))


		f1 = sklearn.metrics.f1_score(self.yt2, self.yp, pos_label=1, average='binary')
		p = sklearn.metrics.precision_score(self.yt2, self.yp, pos_label=1, average='binary')
		r = sklearn.metrics.recall_score(self.yt2, self.yp, pos_label=1, average='binary')
		print('F1 Precion Recall')
		print(f1, p, r)

