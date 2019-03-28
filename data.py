import random
import torch
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer


# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py
class InputExample(object):
	def __init__(self, unique_id, text_a, text_b):
		self.unique_id = unique_id
		self.text_a = text_a
		self.text_b = text_b


class InputFeatures(object):
	def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
		self.unique_id = unique_id
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.input_type_ids = input_type_ids

	def __str__(self):
		s = ''
		s += str(self.unique_id) +'\n' 
		s += ' '.join(self.tokens) + '\n'
		return s

def convert_examples_to_features(examples, seq_length, tokenizer):
	features = []
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if tokens_b:
			_truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
		else:
			if len(tokens_a) > seq_length - 2:
				tokens_a = tokens_a[0:(seq_length - 2)]

		tokens = []
		input_type_ids = []
		tokens.append("[CLS]")
		input_type_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			input_type_ids.append(0)
		tokens.append("[SEP]")
		input_type_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				input_type_ids.append(1)
			tokens.append("[SEP]")
			input_type_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		input_mask = [1] * len(input_ids)
		
		while len(input_ids) < seq_length:
			input_ids.append(0)
			input_mask.append(0)
			input_type_ids.append(0)

		assert len(input_ids) == seq_length
		assert len(input_mask) == seq_length
		assert len(input_type_ids) == seq_length

		# if ex_index < 5:
		# 	print("*** Example ***")
		# 	print("unique_id: %s" % (example.unique_id))
		# 	print("tokens: %s" % " ".join([str(x) for x in tokens]))
		# 	print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		# 	print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		# 	print(
		# 		"input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

		features.append(
			InputFeatures(
				unique_id=example.unique_id,
				tokens=tokens,
				input_ids=input_ids,
				input_mask=input_mask,
				input_type_ids=input_type_ids))
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def read_examples(input_file):
	examples = []
	unique_id = 0
	with open(input_file, "r", encoding='utf-8') as reader:
		while True:
			line = reader.readline()
			if not line:
				break
			line = line.strip()
			text_a = None
			text_b = None
			m = re.match(r"^(.*) \|\|\| (.*)$", line)
			if m is None:
				text_a = line
			else:
				text_a = m.group(1)
				text_b = m.group(2)
			examples.append(
				InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
			unique_id += 1
	return examples


class IntentDset():
	def __init__(self, dataset = 'ATIS', split = 'train', Nc = 10, Ni = 1, n_query = 5):
		self.dset = dataset
		self.Nc = Nc
		self.Ni = Ni
		self.n_query = n_query
		tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

		examples = read_examples('data/'+dataset+'/train/seq.in')
		features = convert_examples_to_features(examples=examples, seq_length=16, tokenizer=tokenizer)

		self.sents = features

		omit = []
		if dataset == 'ATIS':
			omit = ['atis_ground_service#atis_ground_fare','atis_aircraft#atis_flight#atis_flight_no','atis_airline#atis_flight_no','atis_restriction','atis_cheapest']
			omit = set(omit)

		self.labels = open('data/'+dataset+'/'+split+'/label').readlines()
		for i in range(len(self.labels)):
			self.labels[i] = self.labels[i].strip()

		unqiue_labels = set(self.labels)
		unqiue_labels = list(unqiue_labels.difference(omit))

		self.label2id = dict(zip(unqiue_labels,list(range(len(unqiue_labels)))))
		self.id2label = {v: k for k, v in self.label2id.items()}

		self.label_bins = {k:[] for k,_ in self.id2label.items()}
		
		for i in range(len(self.sents)):
			if self.labels[i] in omit:
				continue
			self.label_bins[self.label2id[self.labels[i]]].append(self.sents[i])

		self.n_labels = len(self.label2id)

		# for k in self.label_bins:
		# 	print(self.id2label[k],k,len(self.label_bins[k]))

	def next_batch(self):
		sup_set = random.sample(range(0,self.n_labels),self.Nc)
		# sup_set = list(range(self.n_labels)[:self.Nc])

		batch = {"sup_set_x":[], "sup_set_y":[], "target_x": [], "target_y": []}

		for n,s in enumerate(sup_set):
			idx = random.sample(range(0,len(self.label_bins[s])),self.Ni+self.n_query)
			# idx = list(range(len(self.label_bins[s])))[:self.Ni+self.n_query]
			for j in range(self.Ni):
				i = idx[j]
				batch["sup_set_x"].append(self.label_bins[s][i])

			for j in range(self.n_query):
				i = idx[j+self.Ni]
				batch["target_x"].append(self.label_bins[s][i])

		sup_input_ids = torch.tensor([f.input_ids for f in batch['sup_set_x']], dtype=torch.long)
		sup_input_mask = torch.tensor([f.input_mask for f in batch['sup_set_x']], dtype=torch.long)
		batch["sup_set_x"] = {}
		batch['sup_set_x']['input_ids'] = sup_input_ids
		batch['sup_set_x']['input_mask'] = sup_input_mask

		target_input_ids = torch.tensor([f.input_ids for f in batch['target_x']], dtype=torch.long)
		target_input_mask = torch.tensor([f.input_mask for f in batch['target_x']], dtype=torch.long)
		batch["target_x"] = {}
		batch['target_x']['input_ids'] = target_input_ids
		batch['target_x']['input_mask'] = target_input_mask

		return batch

# idset = IntentDset(dataset = 'SNIPS', Nc=5)
# while(True):
# 	batch = idset.next_batch()
# 	print(batch['sup_set_x']['input_ids'].size())
# 	print(batch['sup_set_x']['input_mask'].size())
# 	print(batch['target_x']['input_ids'].size())
# 	print(batch['target_x']['input_mask'].size())
# 	break