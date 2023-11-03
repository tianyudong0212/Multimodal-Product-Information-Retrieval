from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
from collections import deque

from transformers import T5ForConditionalGeneration, AutoTokenizer

from pytorch_metric_learning.losses import NTXentLoss

from my_dataset import PureFlickr
from data_helper import convert_semanticID_to_text


# load data
pf_dataset = PureFlickr()
pf_dataloader = DataLoader(
    dataset=pf_dataset,
    batch_size=8,
    collate_fn=pf_dataset.collate_fn_4_pf
)
pf_inferloader = DataLoader(
    dataset=pf_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=pf_dataset.collate_fn_4_pf
)

with open('flickr-30k/ids_related/old2new_id_mapper_k5_c30.pkl', 'rb') as f:
    old2new_mapper = pickle.load(f)

# load model
t5_mdl = T5ForConditionalGeneration.from_pretrained('ptms/flan-t5-small').cuda()
t5_mdl.generate
t5_tokenizer = AutoTokenizer.from_pretrained('ptms/flan-t5-small')

# load optimizer and scheduler
adam_optimizer = Adam(params=t5_mdl.parameters(), lr=3e-4)
step_scheduler = StepLR(
    optimizer=adam_optimizer,
    step_size=1,
    gamma=0.7
)

def train_fn(
        dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
        num_epochs,
):
    num_steps_per_epoch = round(len(pf_dataset) // 4 + 0.5)
    num_whole_steps = num_epochs * num_steps_per_epoch

    scroll_loss_deque = deque(maxlen=20)
    trn_step = 0
    for i_epoch in range(1, num_epochs+1):
        for data in dataloader:
            optimizer.zero_grad()
            trn_step += 1
            new_ids, captions, _ = data
            input_ids = tokenizer(captions, padding=True, max_length=64, truncation=True, return_tensors='pt').input_ids.cuda()
            label_ids = tokenizer(new_ids, padding=True, max_length=32, truncation=True, return_tensors='pt').input_ids.cuda()
            
            outputs = model(input_ids=input_ids, labels=label_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            temp_loss = loss.item()
            scroll_loss_deque.append(temp_loss)
            moving_mean_loss = sum(scroll_loss_deque) / len(scroll_loss_deque)

            # output results every 10 steps
            if trn_step % 20 == 0:
                print(f'Epoch:{i_epoch} temploss:{temp_loss:.04} mvloss:{moving_mean_loss:.04} step No.{trn_step - num_steps_per_epoch*(i_epoch-1)}/{num_steps_per_epoch} {trn_step}/{num_whole_steps}')

            # evaluate every 1000 steps
            if trn_step % 1000 == 0:
                eval_fn(pf_inferloader, model, tokenizer)
                model.train()
        scheduler.step()


def contrastive_train(
        dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
        num_epochs,
):
    num_steps_per_epoch = round(len(pf_dataset) // 4 + 0.5)
    num_whole_steps = num_epochs * num_steps_per_epoch

    loss_fn = NTXentLoss(temperature=7e-2)
    scroll_loss_deque = deque(maxlen=20)
    trn_step = 0
    for i_epoch in range(1, num_epochs+1):
        for data in dataloader:
            optimizer.zero_grad()
            trn_step += 1
            new_ids, captions, figures = data
            input_ids = tokenizer(captions, padding=True, max_length=64, truncation=True, return_tensors='pt').input_ids.cuda()
            
            outputs = model(input_ids=input_ids)
            texts = outputs.last_hidden_state.mean(dim=1)
            loss = loss_fn(texts, figures)
            loss.backward()
            optimizer.step()
            temp_loss = loss.item()
            scroll_loss_deque.append(temp_loss)
            moving_mean_loss = sum(scroll_loss_deque) / len(scroll_loss_deque)

            if moving_mean_loss <= 0.1:
                torch.save(model, f"ckp/contrastive_mdl_{moving_mean_loss}.pt")

def eval_fn(
        dataloader,
        model,
        tokenizer,
        top_K=10,
        num_eval=500
):  
    model.eval()
    all_predicts = []
    all_labels = []
    print('start evaluating...')
    data_iter = iter(dataloader)
    for i in trange(num_eval):
        new_ids, captions, _ = next(data_iter)
        all_labels.extend(new_ids)
        for caption in captions:
            input_ids = tokenizer(caption, return_tensors='pt').input_ids.cuda()
            outputs = model.generate(input_ids, max_length=32, num_beams=top_K, num_return_sequences=top_K)
            predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predicts.append(predicts)
    print('start calculating...')
    num_hit = 0
    num_sample = len(all_labels)
    for i in range(len(all_labels)):
        if all_labels[i] in all_predicts[i]:
            num_hit += 1
    hit_score = num_hit / num_sample
    print(f'hit@{top_K}: {hit_score*100:.04}%')
        



train_fn(pf_dataloader, t5_mdl, t5_tokenizer, adam_optimizer, step_scheduler, 6)