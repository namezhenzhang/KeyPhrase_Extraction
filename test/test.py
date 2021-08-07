from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained("bert-base-uncased")

help(model)