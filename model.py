import torch
from torch import nn 
import numpy as np 

from data import IntentDset
from pytorch_pretrained_bert.modeling import BertModel

class ProtNet(nn.Module):
	def __init__(self, n_input = 768, n_output = 128, bert_model = 'bert-large-uncased'):
		super(ProtNet,self).__init__()
		self.bert = BertModel.from_pretrained(bert_model)

	def forward(self, input_ids, input_mask):
		all_hidden_layers,_ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask)
		hn = all_hidden_layers[-1]
		cls_hn = hn[:,0,:]
		return cls_hn

# pn = ProtNet().cuda(3)
# ds = IntentDset()

# while True:
# 	batch = ds.next_batch()
# 	sup_input_ids = batch['sup_set_x']['input_ids'].cuda(3)
# 	sup_input_masks = batch['sup_set_x']['input_mask'].cuda(3)
# 	pn(sup_input_ids,sup_input_masks)
# 	break

