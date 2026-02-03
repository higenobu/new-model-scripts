import torch
from torch.utils.data import Dataset


# データセット処理用のクラス定義
class EconIndicatorDataset(Dataset):
	def __init__(self, dataframe, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.sentence = dataframe.sentence
		self.targets = self.data.label
		self.max_len = max_len

	def __len__(self):
		return len(self.sentence)

	def __getitem__(self, index):
		sentence = str(self.sentence[index])
		sentence = " ".join(sentence.split())

		inputs = self.tokenizer.encode_plus(
			sentence,
			None,
			add_special_tokens=True,
			max_length=self.max_len,
			pad_to_max_length=True,
			return_token_type_ids=True,
			truncation=True
		)
		ids = inputs['input_ids']
		mask = inputs['attention_mask']

		return {
			'input_ids': torch.tensor(ids, dtype=torch.long),
			'attention_mask': torch.tensor(mask, dtype=torch.long),
			'labels': torch.tensor(self.targets[index], dtype=torch.float)
		}
