import torch
import torch.nn as nn
from configs import get_args
import numpy as np
import logger
from transformers import RobertaModel
log = logger.get_logger(__name__)

class Model(torch.nn.Module):
    def __init__(self, args, tokenizer = None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = RobertaModel.from_pretrain(
            "roberta-base",
            return_dict = False)
        
    def forward(self,input_ids,attention_mask, token_type_ids, **karg):
        cur_batchsize = input_ids[0]
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        logits = self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        return logits
def get_model(tokenizer):
    args = get_args()
    model = Model(args, tokenizer)
    # 多cpu并行
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model