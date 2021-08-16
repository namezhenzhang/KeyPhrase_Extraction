from logging import error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import semeval2017_dataset
from tqdm import tqdm
# import os
from transformers import RobertaTokenizerFast, BertTokenizer, BertForMaskedLM, BertModel
# import collections
import heapq

import matplotlib.pyplot as plt
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, prompt_length=2, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.mask_token = tokenizer.mask_token
        # TODO bertmodel ,BertForMaskedLM 有什么区别
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size,
                            self.model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.config.hidden_size, 2))

    def forward(self, input_ids, attention_mask, token_type_ids, labels, mlm_labels, **kargs):

        cur_batchsize = input_ids.shape[0]
        # inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)

        logits = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids).logits

        predict = logits[mlm_labels == 1, :].view(
            cur_batchsize, self.prompt_length, -1)  # .argmax(dim=2)
        return predict


def test(model, dataloader, keyphrases, template_id=0, prompt_length=0, top=1):
    model.eval()
    with torch.no_grad():
        true = 0
        all = 0
        f = open(
            f'src_semeval/swp/template_{template_id}_prompt_{prompt_length}.txt', 'w', encoding="utf-8")
        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):
                # t.set_description("Epoch {}".format(epochs))
                predict = model(**batch)
                batch_size = len(batch['keyphrases_ids'])
                input_ids = batch["input_ids"]
                keyphrases_ids = batch['keyphrases_ids']
                keyphrases_labels = [keyphrases[i] for i in keyphrases_ids]
                attention_mask = batch["attention_mask"]
                prompt_length = model.prompt_length
                # 计算每个输出的top个最大概率滑动窗口作为输出
                for i in range(batch_size):

                    # 滑动窗口长度
                    swp_len = attention_mask[i].sum()-2-(prompt_length-1)
                    swp_pred_prob = []
                    # 每个窗口概率
                    for j in range(1, swp_len+1):
                        swp = input_ids[i][j:j+prompt_length]
                        prod = 1
                        for k, id in enumerate(swp):
                            prod *= predict[i][k][int(id)]
                        swp_pred_prob.append(prod)
                    # swp_pred_prob = torch.Tensor(swp_pred_prob)
                    assert len(swp_pred_prob) == swp_len, "length is wrong."

                    re1 = map(swp_pred_prob.index,
                              heapq.nlargest(top*top, swp_pred_prob))
                    keyphrases_preds_ids = [
                        [input_ids[i][k] for k in range(j+1, j+1+prompt_length)] for j in re1]
                    keyphrases_preds_including_mask = [model.tokenizer.decode(
                        j) for j in keyphrases_preds_ids]

                    notstop = True
                    while notstop:
                        notstop = False
                        for j, s in enumerate(keyphrases_preds_including_mask):
                            if model.mask_token in s:
                                keyphrases_preds_including_mask.pop(j)
                                notstop = True
                    keyphrases_preds = keyphrases_preds_including_mask[0:top if len(
                        keyphrases_preds_including_mask) >= top else len(keyphrases_preds_including_mask)]

                    f.write("keyphrases_preds: "+str(keyphrases_preds)+'\n')
                    f.write("keyphrases_labels: " +
                            str(keyphrases_labels[i])+'\n')
                    is_true = 0
                    for j, kp in enumerate(keyphrases_preds):
                        if len(keyphrases_labels[i]) != 0:
                            if kp in keyphrases_labels[i]:
                                is_true = 1
                    true += is_true
                    if len(keyphrases_labels[i]) != 0:
                        all += 1
                t.update(1)

            # keyphrases_preds = [model.tokenizer.decode(i) for i in predict]
            # keyphrases_labels = [keyphrases[i] for i in keyphrases_ids]

            # for i, kp in enumerate(keyphrases_preds):
            #     if len(keyphrases_labels[i]) != 0:
            #         if kp in keyphrases_labels[i]:
            #             true += 1
            #         all += 1
            # loss = criterion(predict.view(-1, 2), labels.view(-1))
            # t.set_postfix(loss=loss.item())

    print(str(true)+'\n'+str(all)+'\n'+str(true/all)+'\n')
    f.write(str(true)+'\n'+str(all)+'\n'+str(true/all)+'\n')
    f.close()
    return true, all, true/all


def train(model, dataloader, max_epochs=1):
    epochs = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_list = []
    while (epochs < max_epochs):
        tr_loss = 0.0
        global_step = 0
        epochs += 1

        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):
                t.set_description("Epoch {}".format(epochs))
                predict = model(**batch)
                labels = batch['labels']
                # TODO 这样写对吗
                loss = criterion(predict.view(-1, 2), labels.view(-1))
                t.set_postfix(loss=loss.item())
                t.update(1)

                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps
                optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                # tr_loss += loss.item()

                # if (step + 1) % args.gradient_accumulation_steps == 0:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                #     self.optimizer.step()
                #     self.scheduler.step()
                #     self.optimizer_new_token.step()
                #     self.scheduler_new_token.step()
                #     self.model.zero_grad()
                #     global_step += 1
    plt.plot(loss_list)
    plt.savefig("loss.png")
    plt.show()


datadir_train = r"datasets/semeval2017/train/train2"
datadir_train_abs = r"D:\mytsinghua\nlp\KeyPhrase_Extraction\datasets\semeval2017\test\semeval_articles_test"
datadir_test = r"datasets/semeval2017/test/semeval_articles_test"
template_path = r"datasets/semeval2017/templates.txt"
save_model_path = r"model_params"
# prompt_length = 2
# template_id = 1
top = 5
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = Model(tokenizer=tokenizer)
model.cuda()
file = open(f'src_semeval/swp/accu_traindata15.txt', 'w+', encoding="utf-8")
for template_id in [1, 5]:
    for prompt_length in range(1, 4):
        model.prompt_length = prompt_length
        dataset = semeval2017_dataset(tokenizer)
        dataset.load_data_from(datadir_train, template_path,
                               prompt_length=prompt_length, template_id=template_id)
        dataset.cuda()
        keyphrases = dataset.keyphrases
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        true, all, accu = test(model, dataloader, keyphrases,
                               template_id=template_id, prompt_length=prompt_length, top=top)
        file.write("[template_id: "+str(template_id)+']'+"[prompt_length: " +
                   str(prompt_length)+']'+f"[top: {top}]"+f"[true: {true}][all: {all}][accu: {accu}]"+'\n')
file.close()
# model_dict=model.load_state_dict(torch.load(save_model_path))

# train(model,dataloader,max_epochs=10)

# torch.save(model.state_dict(),save_model_path)


print("end")
