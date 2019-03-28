import torch
from data import IntentDset
from model import ProtNet 
from torch import nn, optim 
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

# https://github.com/cyvius96/prototypical-network-pytorch/blob/master/utils.py
def euclidean_metric(a, b):
	n = a.shape[0]
	m = b.shape[0]
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	logits = -((a - b)**2).sum(dim=2)
	return logits

Nc = 10
Ni = 1
Nq = 1

idset = IntentDset(n_query = Nq)
val_dset = IntentDset(dataset = 'SNIPS', Nc = 5, n_query = Nq)

pn = ProtNet().cuda(0)

param_optimizer = list(pn.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
					lr=5e-5,
					warmup=0.1,
					t_total=10000)

criterion = nn.CrossEntropyLoss()

step = 0

while True:
	pn.train()
	step += 1
	# print('gpu_usage',round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
	batch = idset.next_batch()
	sup_set = batch['sup_set_x']
	qry_set = batch['target_x']

	# https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868/2
	# two forwards will link to two different instance wont overwrite the model 
	sup = pn(sup_set['input_ids'].cuda(),sup_set['input_mask'].cuda())
	qry = pn(qry_set['input_ids'].cuda(),qry_set['input_mask'].cuda())


	sup = sup.view(Ni,Nc,-1).mean(0)
	logits = euclidean_metric(qry, sup)

	label = torch.arange(Nc).repeat(Nq).type(torch.LongTensor).cuda()

	loss = criterion(logits, label)


	# print('gpu_usage',round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	if step%1 == 0:
		print('Iteration :',step,"Loss :",float(loss.item()))

	if step%20 == 0:
		pn.eval()
		pn.cuda(3)
		total = 0
		correct = 0
		for i in range(100):
			batch = val_dset.next_batch()
			sup_set = batch['sup_set_x']
			qry_set = batch['target_x']

			sup = pn(sup_set['input_ids'].cuda(3),sup_set['input_mask'].cuda(3))
			qry = pn(qry_set['input_ids'].cuda(3),qry_set['input_mask'].cuda(3))

			sup = sup.view(Ni,5,-1).mean(0)
			logits = euclidean_metric(qry, sup).max(1)[1].cpu()

			label = torch.arange(5).repeat(Nq).type(torch.LongTensor)
			correct += float(torch.sum(logits==label).item())
			total += 5*Ni
		# print(correct,'/',total)
		print('Accuracy :',correct/total)
		pn.cuda(0)
	if step%100000 == 0:
		break