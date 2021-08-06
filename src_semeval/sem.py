import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from transformers import RobertaTokenizerFast,BertTokenizer
class semeval2017_dataset(Dataset):
    def __init__(self,tokenizer):
        self.datasetname = "semeval2017"
        self.tokenizer = tokenizer
        self.dataset = []
        self.dataset_len = 0
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        return self.dataset[index]

    def get_fileNames(self,root, suffix=None):
        if not os.path.isabs(root):
            root = os.path.join(os.getcwd(),root)
        names = os.listdir(root)
        result = []
        if suffix:
            for name in names:
                if os.path.splitext(name)[1] == suffix:
                    result.append(os.path.splitext(name)[0])
        else:
            result = names
        return result
    def load_data_from(self,datadir):
        self.filenames = self.get_fileNames(datadir,".txt")
        self.dataset_len = len(self.filenames)
        origin_data={}
        origin_data["text"]=[]
        origin_data["keyphrase"]=[]
        origin_data["filename"] = []
        flict_pos_num = 0
        not_same_kp = 0
        not_same_log = []
        same_kp = 0
        for filename in tqdm(self.filenames):
            origin_data['filename'].append(filename)
            with open(os.path.join(datadir,filename+".txt"), 'r',encoding='utf-8') as file_txt:
                for line in file_txt:
                    origin_data["text"].append(line.strip())
                    break
            with open(os.path.join(datadir,filename+".ann"), 'r',encoding='utf-8') as file_ann:
                keyphrase_single = []
                for line in file_ann:
                    line = line.strip().split("\t")
                    keyphrase_single.append(line)
                origin_data["keyphrase"].append(keyphrase_single)
        assert self.dataset_len == len(origin_data["text"]) == len( origin_data["keyphrase"]) , "lengths are not same."
        for i in tqdm(range(self.dataset_len)):
            text = origin_data["text"][i]
            filename = origin_data["filename"][i]
            keyphrases = origin_data["keyphrase"][i]
            text_token = tokenizer(text)
            text_len = len(text_token["input_ids"])
            keyphrase_pos = []
            for keyphrase in keyphrases:
                # 暂且只考虑T的关键词，且不考虑分类
                if keyphrase[0][0] is not 'T': 
                    continue
                tmp = keyphrase[1].split(" ")
                keyphrase = keyphrase[2]
                keyphrase_type,start,end = tmp[0], int(tmp[1]), int(tmp[2])
                pretoken = self.tokenizer(text[0:start].strip())
                pretoken_len = len(pretoken["input_ids"])
                kp_token = self.tokenizer(text[start:end].strip())
                kp_token_len = len(kp_token["input_ids"])

                keyphrase_token = [text_token["input_ids"][keyphrase_idx] for keyphrase_idx in range(pretoken_len-1,pretoken_len+kp_token_len-3)]
                keyphrase2 = tokenizer.decode(keyphrase_token)
                keyphrase1 = tokenizer.decode(kp_token["input_ids"][1:-1])
                if keyphrase_token != kp_token["input_ids"][1:-1]:
                    not_same_log.append(filename+':'+keyphrase1+':'+keyphrase2)
                    not_same_kp+=1
                same_kp += 1
                keyphrase_pos.append(list(range(pretoken_len-1,pretoken_len+kp_token_len-3)))

            pos_label = [0] * text_len
            
            for pos_group in keyphrase_pos: 
                for pos in pos_group: 
                    if pos_label[pos] == 1: 
                        flict_pos_num += 1
                    else:
                        pos_label[pos]=1
            text_token['labels'] = pos_label
            self.dataset.append(text_token)
        print("flict_pos_num",flict_pos_num)
        print("not_same_kp",not_same_kp)
        print("same_kp",same_kp)
        print("not_same_log",not_same_log)



datadir_train = r"datasets\semeval2017\train\train2"
datadir_train_abs = r"D:\mytsinghua\nlp\KeyPhrase_Extraction\datasets\semeval2017\test\semeval_articles_test"
datadir_test = r"datasets\semeval2017\test\semeval_articles_test"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",add_prefix_space=True)

dataset = semeval2017_dataset(tokenizer)

dataset.load_data_from(datadir_test)
print("end")