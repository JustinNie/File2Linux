"""
Collect all the functions in this project.
Author: Justin Nie
Date: Apr 28, 2018
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import math
import pickle
import re


MODEL = 2
NUMBER = 5
IN_WORDS = NUMBER
IN_VECTOER = 50
IN_FESTURES = IN_WORDS * IN_VECTOER
HIDDEN_VECTOR = 50
HIDDEN_FEATURES = 5 * HIDDEN_VECTOR
OUT_FEATURES = 50

LR = 0.01
EPOCH = 1000
BATCH_SIZE = 100

torch.manual_seed(10)



def ConstructOriginal():
	'''
		Construct a dictionary to replace wn.synsets() in lemmas
		Input File: None
		Output File: SynsetLemmas.pkl
		'''
	from nltk.corpus import wordnet as wn 

	synset_name_lemmas= {}

	for synset in wn.all_synsets():
		item = synset.name()
		item_dict = synset.lemma_names()

		synset_name_lemmas[item] = item_dict

	f_synset_name_lemmas = open('../Data/SynsetLemmas.pkl', 'wb')
	pickle.dump(synset_name_lemmas, f_synset_name_lemmas)





def CountWordNet():
	'''
		Usage: Count and analyze the basic information about the WordNet.
		Input File: None.
		Output File: LemmaLengthAll.npy, SynsetNameAll.npy.
	'''
	
	f_synset_name_lemmas = open('../Data/SynsetLemmas.pkl', 'rb')
	synset_name_lemmas = pickle.load(f_synset_name_lemmas)

	synset_name = []
	lemma_length = []

	synset_name_number = {}
	synset_number_name = {}
	number = 0
	for item, lemmas in synset_name_lemmas.items():
		name = item
		lemma_length.append(len(lemmas))
		for i in range(int(len(lemmas) / NUMBER) + 1):
			number += 1
			name = name + str(i)

			synset_name.append(name)
			synset_name_number[name] = number
			synset_number_name[number] = name
			name = name[:-1]

	#f1 = open('../Data/SynsetNameNumber.pkl', 'wb')
	#pickle.dump(synset_name_number, f1)

	#f2 = open('../Data/SynsetNumberName.pkl', 'wb')
	#pickle.dump(synset_number_name, f2)

	synset_name.sort()


	synset_name = np.array(synset_name)
	lemma_length = np.array(lemma_length)

	np.save('../Data/Original/SynsetNameAll.npy', synset_name)
	np.save("../Data/Original/LemmaLengthAll.npy", lemma_length)



def ConstructAllDict():
	'''
		Usage: Construct all synsets in WordNet into a dictionary.
		Input File: None.
		Output File: SynsetDictAll.pkl.
	'''
	synset_dict = {}

	f_synset_name_lemmas = open('../Data/SynsetLemmas.pkl', 'rb')
	synset_name_lemmas = pickle.load(f_synset_name_lemmas)

	for synset, lemmas in synset_name_lemmas.items():
		item = synset
		item_dict = lemmas

		index = 0
		while len(item_dict) % NUMBER != 0:
			item_dict.append(item_dict[index])
			index += 1

		item_dict = np.array(item_dict)
		item_dict = item_dict.reshape(-1, NUMBER)

		for i in range(len(item_dict)):
			name = item[:-1]
			name = name + str(i)
			synset_dict[name] = item_dict[i]
			if len(synset_dict[name]) != NUMBER:
				print('Wrong!!!!!!')

	f_dict = open('../Data/Original/SynsetDictAll.pkl', 'wb')
	pickle.dump(synset_dict, f_dict)


def ConstructNewDict():
	'''
		Construct the new dictionary, which stores the synset and lemma names
		according to the vocabulary.
		This is to make sure that all the lemma names could be found in
		vocabulary.

		Input File: vocab.txt
		Output File: SynsetDict.npy, LemmaLength.npy, SynsetNames.npy, 
					 SynsetNameNumber.pkl, SynsetNumberName.pkl.
	'''

	f_synset_name_lemmas = open('../Data/SynsetLemmas.pkl', 'rb')
	synset_name_lemmas = pickle.load(f_synset_name_lemmas)

	raw = open('../Data/vocab.txt', 'r').read()
	word_vocab = re.findall(r'[a-zA-Z]+', raw)

	synset_dict = {}
	synset_names = []
	synset_name_number = {}
	synset_number_name = {}
	total_length = len(synset_name_lemmas)
	length = []
	words = []

	number = 0
	i = 0
	for synset, lemmas in synset_name_lemmas.items():
		i += 1
		if i % 1000 == 0:
			print('Process %.2f%%' % (i / total_length * 100))

		item_dict = lemmas
		name = synset

		new_item_dict = []
		for word in item_dict:
			if word in word_vocab:
				new_item_dict.append(word)
				words.append(word)

		if len(new_item_dict) != 0:
			length.append(len(new_item_dict))
			index = 0
			while len(new_item_dict) % NUMBER != 0:
				new_item_dict.append(new_item_dict[index])
				index += 1

			new_item_dict = np.array(new_item_dict)
			new_item_dict = new_item_dict.reshape(-1, NUMBER)


			for j in range(len(new_item_dict)):
				name = name + str(j)
				synset_dict[name] = new_item_dict[j]
				synset_names.append(name)
				synset_name_number[name] = number
				synset_number_name[number] = name
				name = name[:-1]
				number += 1

	original_length = len(synset_names)

	#Add fake synset
	length_all = len(word_vocab)
	for index, word in enumerate(word_vocab):
		if index % 1000 == 0:
			print('Fake %.2f%%' % (index / length_all * 100))

		if word not in words:
			fake_synset_name = word + '.f.010'
			new_item_dict = [word] * NUMBER

			synset_dict[fake_synset_name] = new_item_dict
			synset_names.append(fake_synset_name)
			synset_name_number[fake_synset_name] = number
			synset_number_name[number] = fake_synset_name
			number += 1

	print(original_length)
	print(len(synset_names))
			
	length = np.array(length)
	np.save('../Data/Original/LemmaLength.npy', length)

	f_dict = open('../Data/Original/SynsetDict.pkl', 'wb')
	pickle.dump(synset_dict, f_dict)

	synset_names = np.array(synset_names)
	np.save('../Data/Original/SynsetNames.npy', synset_names)

	f_number_names = open('../Data/Original/SynsetNumberName.pkl', 'wb')
	pickle.dump(synset_number_name, f_number_names)

	f_name_number = open('../Data/Original/SynsetNameNumber.pkl', 'wb')
	pickle.dump(synset_name_number, f_name_number)


def GetWordVector(vectors_file):
	'''

	'''
	word_vectors = {}
	with open(vectors_file, 'r') as f:
		for line in f:
			values = line.rstrip().split(' ')
			word_vectors[values[0]] = [float(x) for x in values[1:]]

	return word_vectors



def ConstructData(vectors_file):
	'''
		Convert the word dictionary and word vectors into training data.
		Input: vectors.txt, SynsetDict.pkl.
		Output: TrainVectors.npy, TrainSynsets.npy.	
	'''

	word_vectors = GetWordVector(vectors_file)

	f_dict = open('../Data/Original/SynsetDict.pkl', 'rb')
	synset_dict = pickle.load(f_dict)

	f = open('../Data/Original/SynsetNameNumber.pkl', 'rb')
	synset_name_number = pickle.load(f)

	train_vectors = []
	train_synsets = []

	for key, values in synset_dict.items():
		vector = []
		for word in values:
			vector.extend(word_vectors[word])

		train_vectors.append(vector)
		train_synsets.append(synset_name_number[key])

	train_vectors = np.array(train_vectors)
	train_synsets = np.array(train_synsets)

	np.save('../Data/Original/TrainVectors.npy', train_vectors)
	np.save('../Data/Original/TrainSynsets.npy', train_synsets) 


def Train0():
	# import training data
	train_data = np.load('../Data/Original/TrainVectors.npy')
	train_data = torch.FloatTensor(train_data)
	train_label = np.load('../Data/Original/TrainSynsets.npy')
	train_label = torch.IntTensor(train_label)

	dataset = Data.TensorDataset(data_tensor=train_data, target_tensor=train_label)
	data_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

	# print(train_data)
	# print(train_label)

	# Network Definition
	class Encoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(Encoder, self).__init__()
			self.fc1 = nn.Linear(in_features, hidden_features)
			self.fc2 = nn.Linear(hidden_features, out_features)

		def forward(self, x_input):
			x_feature = self.fc1(x_input)
			x_feature_out = F.tanh(x_feature)

			x_out = self.fc2(x_feature_out)

			return x_feature, x_out

	class Decoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(Decoder, self).__init__()
			self.fc3 = nn.Linear(out_features, hidden_features)
			self.fc4 = nn.Linear(hidden_features, in_features)

		def forward(self, x_out):
			x_feature = self.fc3(x_out)
			x_feature_input = F.tanh(x_feature)

			x_input = self.fc4(x_feature_input)

			return x_input

	class AutoEncoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(AutoEncoder, self).__init__()
			self.encoder = Encoder(in_features, hidden_features, out_features)
			self.decoder = Decoder(in_features, hidden_features, out_features)

		def forward(self, x):
			x_feature, encoded = self.encoder(x)
			decoded = self.decoder(encoded)

			return x_feature, encoded, decoded

	autoencoder = AutoEncoder(IN_FESTURES, HIDDEN_FEATURES, OUT_FEATURES)
	# print(autoencoder)

	optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
	loss_func = nn.MSELoss()

	losses = []

	# train
	print('Train the original:')
	for epoch in range(EPOCH):
		for step, (x, y) in enumerate(data_loader):
			batch_x = Variable(x)
			batch_y = Variable(x)

			_, _, decoded = autoencoder(batch_x)

			loss = loss_func(decoded, batch_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.append(loss.data[0])
			if step % 100 == 0:
				print('Epoch: ', epoch, '\tStep: ', step, '\tLoss: %.2f' % loss.data[0])

		torch.save(autoencoder.state_dict(), '../Data/Model0/paras%d.pkl' % epoch)

	plt.plot(range(len(losses)), losses)
	plt.xlabel('Step')
	plt.ylabel('Loss')
	plt.title("Losses' Change with Step 0")
	plt.savefig('../Data/Model0/losses.jpg')
	plt.cla()

	losses = np.array(losses)
	np.save('../Data/Model0/losses.npy', losses)

def Train1():
	# import training data
	train_data = np.load('../Data/Original/TrainVectors.npy')
	train_data = torch.FloatTensor(train_data)
	train_label = np.load('../Data/Original/TrainSynsets.npy')
	train_label = torch.IntTensor(train_label)

	dataset = Data.TensorDataset(data_tensor=train_data, target_tensor=train_label)
	data_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

	# print(train_data)
	# print(train_label)

	# Network Definition

	class JustNet(nn.Module):
		def __init__(self, in_features, out_features, number, bias=True):
			super(JustNet, self).__init__()
			self.in_features = in_features
			self.out_features = out_features
			self.number = number
			self.weight = Parameter(torch.Tensor(out_features, in_features))
			if bias:
				self.bias = Parameter(torch.Tensor(out_features))
			else:
				self.register_parameter('bias', None)
			self.reset_parameters()

		def reset_parameters(self):
			border = 1.0 / math.sqrt(self.in_features)
			new_border = 1.0 / math.sqrt(self.in_features / self.number)

			self.weight.data.uniform_(-border, border)
			if self.bias is not None:
				self.bias.data.uniform_(-border, border)

			in_step = int(self.in_features / self.number)
			out_step = int(self.out_features / self.number)
			for num in range(self.number):
				part_weight = torch.Tensor(out_step, in_step).uniform_(-new_border, new_border)
				for i in range(out_step):
					for j in range(in_step):
						index_i = i + num * out_step
						index_j = j + num * in_step
						self.weight.data[index_i, index_j] = part_weight[i, j]

		def forward(self, x):
			return F.linear(x, self.weight, self.bias)

	class Encoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(Encoder, self).__init__()
			self.fc1 = JustNet(in_features, hidden_features, number)
			self.fc2 = nn.Linear(hidden_features, out_features)

		def forward(self, x_input):
			x_feature = self.fc1(x_input)
			x_feature_out = F.tanh(x_feature)

			x_out = self.fc2(x_feature_out)

			return x_feature, x_out

	class Decoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(Decoder, self).__init__()
			self.fc3 = nn.Linear(out_features, hidden_features)
			self.fc4 = JustNet(hidden_features, in_features, number)

		def forward(self, x_out):
			x_feature = self.fc3(x_out)
			x_feature_input = F.tanh(x_feature)

			x_input = self.fc4(x_feature_input)

			return x_input

	class AutoEncoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(AutoEncoder, self).__init__()
			self.encoder = Encoder(in_features, hidden_features, out_features, number)
			self.decoder = Decoder(in_features, hidden_features, out_features, number)

		def forward(self, x):
			x_feature, encoded = self.encoder(x)
			decoded = self.decoder(encoded)

			return x_feature, encoded, decoded

	autoencoder = AutoEncoder(IN_FESTURES, HIDDEN_FEATURES, OUT_FEATURES, IN_WORDS)
	# print(autoencoder)

	optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
	loss_func = nn.MSELoss()

	losses = []

	# train
	print('Train the first:')
	for epoch in range(EPOCH):
		for step, (x, y) in enumerate(data_loader):
			batch_x = Variable(x)
			batch_y = Variable(x)

			_, _, decoded = autoencoder(batch_x)

			loss = loss_func(decoded, batch_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.append(loss.data[0])
			if step % 100 == 0:
				print('Epoch: ', epoch, '\tStep: ', step, '\tLoss: %.2f' % loss.data[0])

		torch.save(autoencoder.state_dict(), '../Data/Model1/paras%d.pkl' % epoch)

	plt.plot(range(len(losses)), losses)
	plt.xlabel('Step')
	plt.ylabel('Loss')
	plt.title("Losses' Change with Step 1")
	plt.savefig('../Data/Model1/losses.jpg')
	plt.cla()

	losses = np.array(losses)
	np.save('../Data/Model1/losses.npy', losses)


def Save0():
	# import training data
	train_data = np.load('../Data/Original/TrainVectors.npy')
	train_data = torch.FloatTensor(train_data)
	train_label = np.load('../Data/Original/TrainSynsets.npy')
	train_label = torch.IntTensor(train_label)

	# Network Definition
	class Encoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(Encoder, self).__init__()
			self.fc1 = nn.Linear(in_features, hidden_features)
			self.fc2 = nn.Linear(hidden_features, out_features)

		def forward(self, x_input):
			x_feature = self.fc1(x_input)
			x_feature_out = F.tanh(x_feature)

			x_out = self.fc2(x_feature_out)

			return x_feature, x_out

	class Decoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(Decoder, self).__init__()
			self.fc3 = nn.Linear(out_features, hidden_features)
			self.fc4 = nn.Linear(hidden_features, in_features)

		def forward(self, x_out):
			x_feature = self.fc3(x_out)
			x_feature_input = F.tanh(x_feature)

			x_input = self.fc4(x_feature_input)

			return x_input

	class AutoEncoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features):
			super(AutoEncoder, self).__init__()
			self.encoder = Encoder(in_features, hidden_features, out_features)
			self.decoder = Decoder(in_features, hidden_features, out_features)

		def forward(self, x):
			x_feature, encoded = self.encoder(x)
			decoded = self.decoder(encoded)

			return x_feature, encoded, decoded

	# save the vectors
	synset_name_vector = {}
	word_vector = {}

	f = open('../Data/Original/SynsetNumberName.pkl', 'rb')
	synset_number_name = pickle.load(f)

	f = open('../Data/Original/SynsetDict.pkl', 'rb')
	synset_dict = pickle.load(f)

	autoencoder = AutoEncoder(IN_FESTURES, HIDDEN_FEATURES, OUT_FEATURES)

	for epoch in range(EPOCH):
		print('\nProcess %d epoch...' % epoch)
		autoencoder.load_state_dict(torch.load('../Data/Model0/paras%d.pkl' % epoch))

		length = len(train_label)
		print('Create vectors...')
		for i in range(length):
			if i % 1000 == 0:
				# print('Create vector: %.2f%%' %(i/length * 100))
				pass

			x = train_data[i]
			x = Variable(x)
			x_feature, encoded, _ = autoencoder(x)

			x_feature = x_feature.data.numpy()
			encoded = encoded.data.numpy()

			synset_name = synset_number_name[train_label[i]]
			synset_name_vector[synset_name] = encoded

			lemmas = synset_dict[synset_name]

			# x_feature = (x_feature - x_feature.min()) / (x_feature.max() - x_feature.min())
			# x_feature = (x_feature * 4) - 2

			for j, word in enumerate(lemmas):
				word_vector[word] = x_feature[j * HIDDEN_VECTOR: (j + 1) * HIDDEN_VECTOR]

		print('Write the vectors into pickle file...')
		f = open('../Data/Model0/NewWordVector%d.pkl' % epoch, 'wb')
		pickle.dump(word_vector, f)

		f = open('../Data/Model0/SynsetVector%d.pkl' % epoch, 'wb')
		pickle.dump(synset_name_vector, f)

		print('Write the word vectors into text file...')
		f = open('../Data/Model0/NewWordVector%d.txt' % epoch, 'w')
		length = len(word_vector)

		for i, (key, values) in enumerate(word_vector.items()):
			f.write(key + ' ')
			for index, value in enumerate(values):
				f.write(str(value))
				if index < (HIDDEN_VECTOR - 1):
					f.write(' ')
			f.write('\n')

		print('Write the synset vector into text file...')
		f = open('../Data/Model0/SynsetVector%d.txt' % epoch, 'w')
		length = len(synset_name_vector)
		for i, (key, values) in enumerate(synset_name_vector.items()):
			f.write(key + ' ')
			for index, value in enumerate(values):
				f.write(str(value))
				if index < (HIDDEN_VECTOR - 1):
					f.write(' ')
			f.write('\n')


def Save1():
	# import training data
	train_data = np.load('../Data/Original/TrainVectors.npy')
	train_data = torch.FloatTensor(train_data)
	train_label = np.load('../Data/Original/TrainSynsets.npy')
	train_label = torch.IntTensor(train_label)

	# Network Definition

	class JustNet(nn.Module):
		def __init__(self, in_features, out_features, number, bias=True):
			super(JustNet, self).__init__()
			self.in_features = in_features
			self.out_features = out_features
			self.number = number
			self.weight = Parameter(torch.Tensor(out_features, in_features))
			if bias:
				self.bias = Parameter(torch.Tensor(out_features))
			else:
				self.register_parameter('bias', None)
			self.reset_parameters()

		def reset_parameters(self):
			border = 1.0 / math.sqrt(self.in_features)
			new_border = 1.0 / math.sqrt(self.in_features / self.number)

			self.weight.data.uniform_(-border, border)
			if self.bias is not None:
				self.bias.data.uniform_(-border, border)

			in_step = int(self.in_features / self.number)
			out_step = int(self.out_features / self.number)
			for num in range(self.number):
				part_weight = torch.Tensor(out_step, in_step).uniform_(-new_border, new_border)
				for i in range(out_step):
					for j in range(in_step):
						index_i = i + num * out_step
						index_j = j + num * in_step
						self.weight.data[index_i, index_j] = part_weight[i, j]

		def forward(self, x):
			return F.linear(x, self.weight, self.bias)

	class Encoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(Encoder, self).__init__()
			self.fc1 = JustNet(in_features, hidden_features, number)
			self.fc2 = nn.Linear(hidden_features, out_features)

		def forward(self, x_input):
			x_feature = self.fc1(x_input)
			x_feature_out = F.tanh(x_feature)

			x_out = self.fc2(x_feature_out)

			return x_feature, x_out

	class Decoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(Decoder, self).__init__()
			self.fc3 = nn.Linear(out_features, hidden_features)
			self.fc4 = JustNet(hidden_features, in_features, number)

		def forward(self, x_out):
			x_feature = self.fc3(x_out)
			x_feature_input = F.tanh(x_feature)

			x_input = self.fc4(x_feature_input)

			return x_input

	class AutoEncoder(nn.Module):
		"""docstring for Encoder"""

		def __init__(self, in_features, hidden_features, out_features, number):
			super(AutoEncoder, self).__init__()
			self.encoder = Encoder(in_features, hidden_features, out_features, number)
			self.decoder = Decoder(in_features, hidden_features, out_features, number)

		def forward(self, x):
			x_feature, encoded = self.encoder(x)
			decoded = self.decoder(encoded)

			return x_feature, encoded, decoded

	autoencoder = AutoEncoder(IN_FESTURES, HIDDEN_FEATURES, OUT_FEATURES, IN_WORDS)
	# print(autoencoder)

	# save the vectors
	synset_name_vector = {}
	word_vector = {}

	f = open('../Data/Original/SynsetNumberName.pkl', 'rb')
	synset_number_name = pickle.load(f)

	f = open('../Data/Original/SynsetDict.pkl', 'rb')
	synset_dict = pickle.load(f)

	for epoch in range(EPOCH):
		print('\nProcess %d epoch...' % epoch)
		autoencoder.load_state_dict(torch.load('../Data/Model1/paras%d.pkl' % epoch))

		length = len(train_label)
		print('Create vectors...')
		for i in range(length):
			if i % 1000 == 0:
				# print('Create vector: %.2f%%' %(i/length * 100))
				pass

			x = train_data[i]
			x = Variable(x)
			x_feature, encoded, _ = autoencoder(x)

			x_feature = x_feature.data.numpy()
			encoded = encoded.data.numpy()

			synset_name = synset_number_name[train_label[i]]
			synset_name_vector[synset_name] = encoded

			lemmas = synset_dict[synset_name]

			# x_feature = (x_feature - x_feature.min()) / (x_feature.max() - x_feature.min())
			# x_feature = (x_feature * 4) - 2

			for j, word in enumerate(lemmas):
				word_vector[word] = x_feature[j * HIDDEN_VECTOR: (j + 1) * HIDDEN_VECTOR]

		print('Write the vectors into pickle file...')
		f = open('../Data/Model1/NewWordVector%d.pkl' % epoch, 'wb')
		pickle.dump(word_vector, f)

		f = open('../Data/Model1/SynsetVector%d.pkl' % epoch, 'wb')
		pickle.dump(synset_name_vector, f)

		print('Write the word vectors into text file...')
		f = open('../Data/Model1/NewWordVector%d.txt' % epoch, 'w')
		length = len(word_vector)

		for i, (key, values) in enumerate(word_vector.items()):
			f.write(key + ' ')
			for index, value in enumerate(values):
				f.write(str(value))
				if index < (HIDDEN_VECTOR - 1):
					f.write(' ')
			f.write('\n')

		print('Write the synset vector into text file...')
		f = open('../Data/Model1/SynsetVector%d.txt' % epoch, 'w')
		length = len(synset_name_vector)
		for i, (key, values) in enumerate(synset_name_vector.items()):
			f.write(key + ' ')
			for index, value in enumerate(values):
				f.write(str(value))
				if index < (HIDDEN_VECTOR - 1):
					f.write(' ')
			f.write('\n')


def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = '../Question/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

    accuracy = correct_tot / float(count_tot)

    return accuracy

def Evaluate(model, epoch):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='../Data/vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='../Data/Model%d/NewWordVector%d.txt' %(model, epoch), type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}


    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    accuracy = evaluate_vectors(W_norm, vocab, ivocab)

    return accuracy

def GetResults():
	for model in range(MODEL):
		print('Evaluate model %d...' %model)
		results = []
		for epoch in range(EPOCH):
			accuracy = Evaluate(model, epoch)
			results.append(accuracy)

		results = np.array(results)
		np.save('../Data/Model%d/results.npy' %model, results)

def PrintResults():
	figure_index = 0
	for model in range(MODEL):
		figure_index += 1
		results = np.load('../Data/Model%d/results.npy' %model)
		plt.figure(figure_index)
		plt.plot(results, label='Model%d' %model)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig('../Data/Model%d/results.jpg' %model)
		plt.cla()



#print('Construct original wordnet with synset name and synset lemmas...')
#ConstructOriginal()

print('Count the basic information of wordnet...')
CountWordNet()
print('Construct dictionary of all synsets...')
ConstructAllDict()
print('Construct dictionary of new synsets according to the vocabulary...')
ConstructNewDict()

ConstructData('../Data/vectors.txt')

print('Training model 0...')
Train0()
print('Training model 1...')
Train1()

print('\n\nSave vectors of model 0...')
Save0()
print('\n\nSave vectors of model 1...')
Save1()

print('\n\nEvaluate...')
GetResults()


PrintResults()


