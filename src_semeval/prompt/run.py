from logging import error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from transformers import RobertaTokenizerFast, BertTokenizer, BertForMaskedLM, BertModel
import collections

import matplotlib.pyplot as plt
import numpy as np


class semeval2017_dataset(Dataset):
    def __init__(self, tokenizer):
        self.datasetname = "semeval2017"
        self.tokenizer = tokenizer
        self.dataset = {"input_ids": [], "attention_mask": [],
                        "token_type_ids": [], "mlm_labels": [], "labels": []}
        self.dataset_len = 0

    def __len__(self):
        return len(self.dataset["labels"])

    def __getitem__(self, index):
        return {key: data[index] for key, data in self.dataset.items()}

    def cuda(self):
        for key in self.dataset:
            self.dataset[key] = self.dataset[key].cuda()

    def get_fileNames(self, root, suffix=None):
        if not os.path.isabs(root):
            root = os.path.join(os.getcwd(), root)
        names = os.listdir(root)
        result = []
        if suffix:
            for name in names:
                if os.path.splitext(name)[1] == suffix:
                    result.append(os.path.splitext(name)[0])
        else:
            result = names
        return result

    def load_data_from(self, datadir, template_path, max_length=512, prompt_length=1):
        '''
        load data from datadir
        '''
        self.filenames = self.get_fileNames(datadir, ".txt")
        self.dataset_len = len(self.filenames)

        # 统计量
        flict_pos_num = 0
        not_same_kp = 0
        not_same_log = []
        same_kp = 0
        error_T = 0
        kp_len_dict = collections.defaultdict(int)

        self.dataset_without_prompt = []

        # 原始数据
        origin_data = {}
        origin_data["text"] = []
        origin_data["keyphrase"] = []
        origin_data["filename"] = []
        for filename in self.filenames:
            origin_data['filename'].append(filename)
            with open(os.path.join(datadir, filename+".txt"), 'r', encoding='utf-8') as file_txt:
                for line in file_txt:
                    origin_data["text"].append(line.strip())
                    break
            with open(os.path.join(datadir, filename+".ann"), 'r', encoding='utf-8') as file_ann:
                keyphrase_single = []
                for line in file_ann:
                    line = line.strip().split("\t")
                    keyphrase_single.append(line)
                origin_data["keyphrase"].append(keyphrase_single)

        assert self.dataset_len == len(origin_data["text"]) == len(
            origin_data["keyphrase"]), "lengths are not same."

        for i in tqdm(range(self.dataset_len)):
            text = origin_data["text"][i]
            filename = origin_data["filename"][i]
            keyphrases = origin_data["keyphrase"][i]
            text_token = self.tokenizer(text)
            text_len = len(text_token["input_ids"])
            keyphrase_pos = []

            # 提取文中的关键短语和位置
            for keyphrase in keyphrases:
                # 暂且只考虑T的关键词，且不考虑分类
                if keyphrase[0][0] != 'T':
                    continue
                tmp = keyphrase[1].split(" ")
                keyphrase = keyphrase[2]
                try:
                    keyphrase_type, start, end = tmp[0], int(
                        tmp[1]), int(tmp[2])
                except:
                    error_T += 1
                    continue

                pretoken = self.tokenizer(text[0:start].strip())
                pretoken_len = len(pretoken["input_ids"])
                kp_token = self.tokenizer(text[start:end].strip())
                kp_token_len = len(kp_token["input_ids"])
                kp_len_dict[kp_token_len-2] += 1
                if kp_token_len-2 != prompt_length:
                    continue

                keyphrase_token = [text_token["input_ids"][keyphrase_idx] for keyphrase_idx in range(
                    pretoken_len-1, pretoken_len+kp_token_len-3)]
                keyphrase2 = self.tokenizer.decode(keyphrase_token)
                keyphrase1 = self.tokenizer.decode(kp_token["input_ids"][1:-1])
                if keyphrase_token != kp_token["input_ids"][1:-1]:
                    not_same_log.append(filename+':'+keyphrase1+':'+keyphrase2)
                    not_same_kp += 1
                same_kp += 1
                keyphrase_pos.append(
                    list(range(pretoken_len-1, pretoken_len+kp_token_len-3)))

            pos_label = [0] * text_len

            for pos_group in keyphrase_pos:
                for pos in pos_group:
                    if pos_label[pos] == 1:
                        flict_pos_num += 1
                    else:
                        pos_label[pos] = 1
            text_token['labels'] = pos_label
            self.dataset_without_prompt.append(text_token)

        print("flict_pos_num", flict_pos_num)
        print("not_same_kp", not_same_kp)
        print("same_kp", same_kp)
        print("not_same_log", not_same_log)
        print("error_T", error_T)

        print("kp_len_dict", sorted(
            kp_len_dict.items(), key=lambda item: item[0]))

        self.template_padding(
            template_path, prompt_length, max_length=max_length)

    def template_padding(self, template_path, prompt_length, max_length=512, template_id=0):
        '''
        attach the template to sentence, then do padding
        '''
        with open(template_path, 'r', encoding='utf-8') as f:
            templates = [line.strip().split() for line in f]
            self.template = templates[template_id]

        tmp = []
        for idx, x in enumerate(self.template):
            if x == "<mask>":
                tmp += [self.tokenizer.mask_token_id]*prompt_length
            elif x == "<text>":
                pass
            else:
                tmp += self.tokenizer(x)["input_ids"][1:-1]
        tmp_len = len(tmp)+2
        assert tmp_len <= max_length, "max_length too short"

        for data in self.dataset_without_prompt:
            composed = {"input_ids": [], "attention_mask": [],
                        "token_type_ids": [], "labels": []}
            composed["input_ids"].append(data["input_ids"][0])
            composed["attention_mask"].append(data["attention_mask"][0])
            composed["token_type_ids"].append(data["token_type_ids"][0])
            composed["labels"].append(0)

            for idx, x in enumerate(self.template):
                if x == "<mask>":
                    composed["input_ids"] += [self.tokenizer.mask_token_id] * \
                        prompt_length
                    composed["attention_mask"] += [1]*prompt_length
                    composed["token_type_ids"] += [composed["token_type_ids"]
                                                   [0]]*prompt_length
                    # TODO 这里可能用1会更好，因为这个mask和keyphrase相关度很高
                    composed["labels"] += [1]*prompt_length
                elif x == "<text>":
                    len_ = len(data["attention_mask"][1:-1])
                    if len_+tmp_len > max_length:
                        len_ = max_length-tmp_len

                    composed["input_ids"] += (data["input_ids"][1:-1])[0:len_]
                    composed["attention_mask"] += [1]*len_
                    composed["token_type_ids"] += [composed["token_type_ids"][0]]*len_
                    composed["labels"] += (data["labels"][1:-1])[0:len_]
                else:
                    token_x = self.tokenizer(x)["input_ids"][1:-1]
                    token_x_len = len(token_x)
                    composed["input_ids"] += token_x
                    composed["attention_mask"] += [1]*token_x_len
                    composed["token_type_ids"] += [composed["token_type_ids"]
                                                   [0]]*token_x_len
                    composed["labels"] += [0]*token_x_len

            composed["input_ids"].append(data["input_ids"][-1])
            composed["attention_mask"].append(data["attention_mask"][-1])
            composed["token_type_ids"].append(data["token_type_ids"][-1])
            composed["labels"].append(0)
            assert len(composed["input_ids"]) == len(composed["attention_mask"]) == len(
                composed["token_type_ids"]) == len(composed["labels"]), "length does not match"

            padding_length = max_length - len(composed["attention_mask"])
            if padding_length > 0:
                composed["input_ids"] += [self.tokenizer.pad_token_id] * \
                    padding_length
                composed["attention_mask"] += [0]*padding_length
                if data["token_type_ids"][-1] == 0:
                    a_m = 1
                else:
                    a_m = 0
                composed["token_type_ids"] += [a_m]*padding_length
                composed["labels"] += [0]*padding_length

            composed["mlm_labels"] = self.get_mask_positions(
                composed["input_ids"])
            assert len(composed["mlm_labels"]) == len(composed["input_ids"]) == len(composed["attention_mask"]) == len(
                composed["token_type_ids"]) == len(composed["labels"]) == max_length, "length does not match"

            self.dataset["input_ids"].append(composed["input_ids"])
            self.dataset["attention_mask"].append(composed["attention_mask"])
            self.dataset["token_type_ids"].append(composed["token_type_ids"])
            self.dataset["labels"].append(composed["labels"])
            self.dataset["mlm_labels"].append(composed["mlm_labels"])

        self.dataset["input_ids"] = torch.Tensor(
            self.dataset["input_ids"]).long()
        self.dataset["attention_mask"] = torch.Tensor(
            self.dataset["attention_mask"]).long()
        self.dataset["token_type_ids"] = torch.Tensor(
            self.dataset["token_type_ids"]).long()
        self.dataset["labels"] = torch.Tensor(self.dataset["labels"]).long()
        self.dataset["mlm_labels"] = torch.Tensor(
            self.dataset["mlm_labels"]).long()

    def get_mask_positions(self, input_ids):
        # label_idx = input_ids.index(self.tokenizer.mask_token_id)
        labels = [-1] * len(input_ids)
        for i,ids in enumerate(input_ids):
            if ids == self.tokenizer.mask_token_id:
                labels[i] = 1
        return labels


class Model(torch.nn.Module):
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        # TODO bertmodel ,BertForMaskedLM 有什么区别
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size,
                            self.model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.config.hidden_size, 2))

    def forward(self, input_ids, attention_mask, token_type_ids, labels, mlm_labels):
        cur_batchsize = input_ids[0]
        # inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        logits = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)[0]
        predict = self.mlp(logits)
        return predict


def get_criterion():
    pass


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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = semeval2017_dataset(tokenizer)
dataset.load_data_from(datadir_train, template_path, prompt_length=2)
dataset.cuda()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model_dict=model.load_state_dict(torch.load(save_model_path))
# model = Model(tokenizer=tokenizer)
# model.cuda()
# train(model,dataloader,max_epochs=10)
# torch.save(model.state_dict(),save_model_path)


print("end")
