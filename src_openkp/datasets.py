from torch.utils.data import Dataset
import logger
import os
import jsonlines
from tqdm import tqdm
import torch
log = logger.get_logger(__name__)


class OpenKPDataset(Dataset):

    def __init__(self,args, tokenizer,split):
        # super(OpenKPDataset,self).__init__()
        self.dirname = 'openkp'
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.tensors = None
        self.get_template()
        self.get_tensors_from_csv(split=split)

    def get_tensors_from_csv(self, split='train'):

        
        texts = []
        with open("datasets\\openkp\\OpenKPEval.jsonl", "r+", encoding="utf8") as f:
            for item in tqdm(jsonlines.Reader(f)):
                texts.append({'url':item['url'],'text': item['text'],'KeyPhrases':item['KeyPhrases']})
        self.u_k_res,self.tensors = self.list2tensor(texts)

    def tokenize(self, data):
        url = data['url']
        text = data['text']
        KeyPhrases = data['KeyPhrases']
        composed = []
        text_idx = None
        for idx,x in enumerate(self.temps['text']):
            if x == "<mask>":
                composed.append(self.tokenizer.mask_token)
            elif x == "<text>":
                composed.append(text)
                text_idx = idx
            else:
                composed.append(x)
        composed = [self.tokenizer.encode(x, add_special_tokens=False) for x in composed]

        lens = [len(x) for x in composed]
        total_len = sum(lens)
        lens[text_idx] = 0
        other_len = sum(lens)+self.tokenizer.num_special_tokens_to_add(False)
        if total_len>self.args.max_seq_length-self.tokenizer.num_special_tokens_to_add(False):
            composed[text_idx] = composed[text_idx][0:self.args.max_seq_length-other_len]

        tokens = [token_id for part in composed for token_id in part]
        assert  len(tokens)<=self.args.max_seq_length-2,"not 510"
        #<cls> text <sep>
        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens)
        assert len(input_ids)<=self.args.max_seq_length,"not 512"
        #[00000,11111]
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens)

        return input_ids, token_type_ids
    def get_mask_positions(self, input_ids):
        '''
        除了mask的位置为1，其余位置为-1
        '''
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels
    def list2tensor(self, data):
        res = {}
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['KeyPhrases'] = []
        res['url'] = []
        res['mlm_labels'] = []

        for idx, i in enumerate(tqdm(data[0:100])):
            input_ids, token_type_ids = self.tokenize(i)

            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)
            if padding_length < 0:
                # print(i)
                raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            mlm_labels = self.get_mask_positions(input_ids)
            res['input_ids'].append(input_ids)
            res['mlm_labels'].append(mlm_labels)
            res['attention_mask'].append(attention_mask)
            res['token_type_ids'].append(token_type_ids)
            res['KeyPhrases'].append(i['KeyPhrases'])
            res['url'].append(i['url'])
        tensor_res = {}
        tensor_res['input_ids'] = torch.Tensor(res['input_ids']).long()
        tensor_res['mlm_labels'] = torch.Tensor(res['mlm_labels']).long()
        tensor_res['attention_mask'] = torch.Tensor(res['attention_mask']).long()
        tensor_res['token_type_ids'] = torch.Tensor(res['token_type_ids']).long()
        u_k_res = {}
        tensor_res['KeyPhrases'] = torch.Tensor(range(len(res['KeyPhrases'])))
        u_k_res['KeyPhrases'] = res['KeyPhrases']
        tensor_res['url'] = torch.Tensor(range(len(res['url'])))
        u_k_res['url'] = res['url']
        log.info("keys : {} ".format(tensor_res.keys))
        return u_k_res,tensor_res

    def __getitem__(self, index):
        return self.tensors[index]
         
    def __len__(self):
        return len(self.tensors)

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()

    def get_mask_positions(self, input_ids):
        '''
        除了mask的位置为1，其余位置为-1
        '''
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def get_template(self, template_id=0):
        temps = {}
        # template.txt
        template_file = open(os.path.join(
            self.args.data_dir, self.dirname, self.args.template_file_name), 'r')
        templates = [line.strip().split() for line in template_file]
        template_id = self.args.template_id
        template = templates[template_id]
        temps['text'] = template
        self.temps = temps
        log.debug(self.temps)

# Datasets = {
#     "openkp":OpenKPDataset
# }