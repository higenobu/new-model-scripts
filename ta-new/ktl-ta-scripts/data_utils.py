import argparse
import json
import re
import pandas as pd
import glob
from tqdm import tqdm
from pyknp import Juman
import mojimoji
from sklearn.model_selection import train_test_split

# Permanent variables
# NEW_FOLDER_PATH = '/home/alkalinemoe/psych_model_scripts/data/new/news10000_final_data'
OLD_FOLDER_PATH = '/home/alkalinemoe/psych_model_scripts/data/old'

# regex
REGEX_PATTERNS = (
	None,
	re.compile(r'^(.+) - 1.[\s\t]*喜びを感じた$'),
	re.compile(r'^(.+) - 2.[\s\t]*恐怖を感じた$'),
	re.compile(r'^(.+) - 3.[\s\t]*驚きを感じた$'),
	re.compile(r'^(.+) - 4.[\s\t]*信頼できる情報と感じた$'),
	re.compile(r'^(.+) - 5.[\s\t]*曖昧な情報と感じた$'),
	re.compile(r'^(.+) - 6.[\s\t]*何かの意図をもって書かれたと感じた$'),
	re.compile(r'^(.+) - 7.[\s\t]*経済に期待がもてると感じた$'),
)

def read_data_from_jsonl(file_path: str) -> tuple:
	"""
	Reads data from a JSONL file.

	Args:
		file_path (str): The path to the JSONL file.

	Returns:
		sentences (list): A list of sentences.
		labels (list): A list of labels corresponding to the sentences.
	"""

	sentences, labels = [], []
	with open(file_path, 'r') as f:
		for line in f:
			json_data = json.loads(line)
			sentence = json_data['sentence']
			label = json_data['label']
			sentences.append(sentence)
			labels.append(label)

	return sentences, labels

def old_training_data_to_list():
	"""
	Reads old training data from JSON files and returns the data as lists.

	Returns:
		sentences_train_old (list): A list of sentences for training.
		labels_train_old (list): A list of labels corresponding to the training sentences.
		sentences_dev_old (list): A list of sentences for development.
		labels_dev_old (list): A list of labels corresponding to the development sentences.
		sentences_test_old (list): A list of sentences for testing.
		labels_test_old (list): A list of labels corresponding to the testing sentences.
	"""

	# Read old training data
	sentences_train_old = [[] for i in range(0, 8)]
	labels_train_old = [[] for i in range(0, 8)]
	for i in range(1, 8):
		with open(f'{OLD_FOLDER_PATH}/train_data_{i}.json', 'r') as f:
			for line in f:
				try:
					json_data = json.loads(line)
					sentence = json_data['sentence']
					# sentence=re.sub(r"\s", "", sentence)
					label = json_data['label']
					sentences_train_old[i].append(sentence)
					labels_train_old[i].append(label)
				except json.JSONDecodeError:
					continue

	# Read old dev data
	sentences_dev_old = [[] for i in range(0, 8)]
	labels_dev_old = [[] for i in range(0, 8)]
	for i in range(1, 8):
		with open(f'{OLD_FOLDER_PATH}/dev_data_{i}.json', 'r') as f:
			for line in f:
				try:
					json_data = json.loads(line)
					sentence = json_data['sentence']
					# sentence=re.sub(r"\s", "", sentence)
					label = json_data['label']
					sentences_dev_old[i].append(sentence)
					labels_dev_old[i].append(label)
				except json.JSONDecodeError:
					continue

	# Read old test data
	sentences_test_old = [[] for i in range(0, 8)]
	labels_test_old = [[] for i in range(0, 8)]
	for i in range(1, 8):
		with open(f'{OLD_FOLDER_PATH}/test_data_{i}.json', 'r') as f:
			for line in f:
				try:
					json_data = json.loads(line)
					sentence = json_data['sentence']
					# sentence=re.sub(r"\s", "", sentence)
					label = json_data['label']
					sentences_test_old[i].append(sentence)
					labels_test_old[i].append(label)
				except json.JSONDecodeError:
					continue

	return sentences_train_old, labels_train_old, sentences_dev_old, labels_dev_old, sentences_test_old, labels_test_old


def calculate_statistics(dataframe, column_range):
	"""
	指定された列の平均値と標準偏差を計算する。

	Args:
		dataframe (pandas.DataFrame): 統計量を計算するデータフレーム。
		column_range (slice): 統計量を計算する列の範囲。

	Returns:
		pandas.DataFrame: 統計量を計算したデータフレーム。
	"""

	dataframe['mean'] = dataframe.iloc[:, column_range].astype(float).mean(axis=1)
	dataframe['sd'] = dataframe.iloc[:, column_range].astype(float).std(axis=1, ddof=0)

	return dataframe

def transform_dataframe(df):
	"""
	読み込んだDataFrameの前処理を行う関数

	Args:
		df (pandas.DataFrame): 前処理を行うデータフレーム。

	Returns:
		pandas.DataFrame: 前処理を行ったデータフレーム。
	"""

	threshold_duration = 1200
	# 数値を抽出するための正規表現パターン
	pattern = r'(\d+)'
	# group(1)のみを返す関数
	repl = lambda m: m.group(1)

	# 数値部分を抽出して新しい列を作成
	# Duration列から数値を抽出し、新しい列に格納

	df['Duration_numeric'] = df['Duration (in seconds)'].str.extract(pattern, expand=False).astype(float)

	# 特定の秒数未満の行を削除
	df = df[(df['Duration_numeric'] > threshold_duration)|(df['Duration_numeric'].isna())]


	data=df.T.iloc[17:,:]
	data = data[data.index!= 'Duration_numeric']
	data['ids'] = data.index.str.split(r'_', expand=False)
	# 新しい列にYの値を格納
	data['id_range'] = data['ids'].apply(lambda s: s[0])
	data["id"]=data["id_range"]
	data['seq'] = data['ids'].apply(lambda s: s[1])
	data['text'] = data[0]

	for i in range(1, 8):
			data['text'] = data['text'].str.replace(REGEX_PATTERNS[i], repl, regex=True)

	return data


def abnormal_alert(dataframes):
	for i,df in enumerate(dataframes):
		if df.shape[0]!=7:
			print("***alart***")
			print(df.shape)
			print(f"{i}番目のdfのshapeが異常です")
			print("***alart***")
		else:
			pass


def process_text(text):
	"""
	テキストの前処理を行う関数

	Args:
		text (str): 前処理を行うテキスト。

	Returns:
		str: 前処理を行ったテキスト。
	"""

	# 形態素解析器のインスタンスを作成
	jumanpp = Juman()
	# テキストを全角に変換
	text = mojimoji.han_to_zen(text)

	# 【】で囲まれている部分を削除
	#text = re.sub('【.*?】', '', text)

	# テキストを文単位に分割
	sentences = re.split('(?<=[。])|\n', text)
	# 文が20文字未満のものを除外
	sentences = [sentence for sentence in sentences if len(sentence) >= 20]

	# 各文に形態素解析を適用し、全体のトークンを集める
	all_tokens = []
	for sentence in sentences:
		tokens = [mrph.midasi for mrph in jumanpp.analysis(sentence.replace('^', '＾')).mrph_list()]
		all_tokens.extend(tokens)

	#Cut-off at 128 tokens
	if len(all_tokens) > 128:
		all_tokens = all_tokens[:128]

	return ' '.join(all_tokens)


def final_data_formatting(NEW_FOLDER_PATH):
	"""
	新しいデータを読み込み、前処理を行う関数

	Args:
		NEW_FOLDER_PATH (str): 新しいデータのフォルダのパス。

	Returns:
		sentences_train_new (list): 訓練データの文章のリスト。
		labels_train_new (list): 訓練データのラベルのリスト。
		sentences_dev_new (list): 検証データの文章のリスト。
		labels_dev_new (list): 検証データのラベルのリスト。
		sentences_test_new (list): テストデータの文章のリスト。
		labels_test_new (list): テストデータのラベルのリスト。
	"""

	# tqdmをPandasに組み込む
	tqdm.pandas()

	# CSVファイルを読み込む
	csv_files = glob.glob(f"{NEW_FOLDER_PATH}/*.csv")
	dataframes = [pd.read_csv(open(file, encoding='Shift-JIS', errors="replace")) for file in csv_files]

	#すべてのshapeを確認し、異常値の時にアラート
	# abnormal_alert(dataframes)

	#読み込んだすべてのdfにtransform_dataframe関数を適用させ、新しいdataframeのリストを作成
	transformed_dataframes = [transform_dataframe(df) for df in dataframes]

	combined_df = pd.concat(transformed_dataframes, ignore_index=True)

	#通常12,13行目が欠損値なはずだが、以下のdataframeのみ存在している？
	# COLUMN_INDEX_TO_CHECK = 12
	# 特定の列インデックスに欠損データがある行を削除する。
	# non_missing_data = combined_df[combined_df.iloc[:, COLUMN_INDEX_TO_CHECK].notnull()]

	# データフレームから指定された数の列だけを保持する。
	NUM_COLUMNS_TO_KEEP = 12
	combined_df=combined_df.iloc[:,:NUM_COLUMNS_TO_KEEP]

	# 簡単な統計量を表示
	COLUMN_RANGE_FOR_STATS = slice(2, 7)
	combined_df = calculate_statistics(combined_df, COLUMN_RANGE_FOR_STATS)

	# 軽く前処理を行う
	COLUMNS_TO_KEEP = [0, 2, 3, 4, 5, 6, 'mean', 'sd', 'id', 'seq', 'text']
	SD_THRESHOLD = 1.4

	combined_df_formatted=combined_df[COLUMNS_TO_KEEP]
	combined_df_formatted=combined_df_formatted[combined_df_formatted["sd"]<SD_THRESHOLD]

	# missing values check
	# missing_values = combined_df_formatted.isnull().sum()
	# print(missing_values)

	# seq列の値を修正
	df = combined_df_formatted

	# "seq"列内で "X.1" を "X" に置換する
	for num in range(1, 8):
		df["seq"] = df["seq"].replace(f"{num}.1", str(num))

	# remove_corona_rows関数を適用し、新しいデータフレームを取得
	# df = remove_corona_rows(df)

	df['text'] = df['text'].progress_apply(process_text)

	#seqの値によって、dfを分解
	combined_df_seq=[[]for _ in range(8)]
	for i in range(1,8):
		combined_df_seq[i] = df[df["seq"]==f"{i}"]

	sentences_train_new=[[]for _ in range(8)]
	labels_train_new=[[]for _ in range(8)]
	sentences_dev_new=[[]for _ in range(8)]
	labels_dev_new=[[]for _ in range(8)]
	sentences_test_new=[[]for _ in range(8)]
	labels_test_new=[[]for _ in range(8)]


	for i in range(1,8):

		# 訓練データ、検証データ、テストデータに分割する
		train_val, test = train_test_split(combined_df_seq[i], test_size=0.1, random_state=42)
		train, val = train_test_split(train_val, test_size=0.111, random_state=42)

		# 訓練データ、検証データ、テストデータのサイズを表示する
		print("カテゴリー：", i)
		print(f"訓練データのサイズ: {len(train)}, 検証データのサイズ: {len(val)}, テストデータのサイズ: {len(test)}")
		print("====================================")

		for _, row in train.iterrows():
			if row["sd"]>=1.0:
				continue
			sentences_train_new[i].append(row['text'])
			labels_train_new[i].append(row['mean'])
		for _, row in val.iterrows():
			#if row["sd"]>=0.75:
			#    continue
			sentences_dev_new[i].append(row['text'])
			labels_dev_new[i].append(row['mean'])
		for _, row in test.iterrows():
			sentences_test_new[i].append(row['text'])
			labels_test_new[i].append(row['mean'])

	return sentences_train_new, labels_train_new, sentences_dev_new, labels_dev_new, sentences_test_new, labels_test_new

def combine_lists(new_data, old_data):
	"""
	Combine the new and old data.

	Args:
		new_data (list): The new data.
		old_data (list): The old data.

	Returns:
		list: The combined data.
	"""

	return [new + old for new, old in zip(new_data, old_data)]

def remove_corona_rows(df):
	print(df.shape)
	new_rows = []

	for index, row in tqdm(df.iterrows()):
		text = row["sentence"]

		# "コロナ"という単語が含まれ、かつ"コロナビール"という単語が含まれない場合のみ新しい行に追加します
		if "コロナ" not in text or "コロナビール"  in text:
			#print(text)
			new_rows.append(row)

	new_df = pd.DataFrame(new_rows)
	new_df.reset_index(drop=True, inplace=True)
	print(new_df.shape)

	return new_df


def process_data_pipeline(NEW_FOLDER_PATH, generate_formatted_data=False):

	if not generate_formatted_data:

		# read formatted data
		sentences_train_combined = [[] for i in range(0, 8)]
		labels_train_combined = [[] for i in range(0, 8)]
		sentences_dev_combined = [[] for i in range(0, 8)]
		labels_dev_combined = [[] for i in range(0, 8)]
		sentences_test_combined = [[] for i in range(0, 8)]
		labels_test_combined = [[] for i in range(0, 8)]

		for i in range(1, 8):
			sentences_train_combined[i], labels_train_combined[i] = read_data_from_jsonl(f"{NEW_FOLDER_PATH}/train_{i}.json")
			sentences_dev_combined[i], labels_dev_combined[i] = read_data_from_jsonl(f"{NEW_FOLDER_PATH}/dev_{i}.json")
			sentences_test_combined[i], labels_test_combined[i] = read_data_from_jsonl(f"{NEW_FOLDER_PATH}/test_{i}.json")

	# Need to generate from the raw files
	else:

		# generally not required
		# old training data
		sentences_train_old, labels_train_old, sentences_dev_old, labels_dev_old, sentences_test_old, labels_test_old = old_training_data_to_list()

		# new training data
		sentences_train_new, labels_train_new, sentences_dev_new, labels_dev_new, sentences_test_new, labels_test_new = final_data_formatting(NEW_FOLDER_PATH)

		# pack data
		sentences_train_combined = combine_lists(sentences_train_new, sentences_train_old)
		labels_train_combined = combine_lists(labels_train_new, labels_train_old)
		sentences_dev_combined = combine_lists(sentences_dev_new, sentences_dev_old)
		labels_dev_combined = combine_lists(labels_dev_new, labels_dev_old)
		sentences_test_combined = combine_lists(sentences_test_new, sentences_test_old)
		labels_test_combined = combine_lists(labels_test_new, labels_test_old)

		# generate formatted jsonl files for training

		for i in range(1, 8):
			# to jsonl
			with open(f"{NEW_FOLDER_PATH}/train_{i}.json", "w") as f:
				for sentence, label in zip(sentences_train_combined[i], labels_train_combined[i]):
					f.write(json.dumps({"sentence": sentence, "label": label}, ensure_ascii=False) + "\n")

			with open(f"{NEW_FOLDER_PATH}/dev_{i}.json", "w") as f:
				for sentence, label in zip(sentences_dev_combined[i], labels_dev_combined[i]):
					f.write(json.dumps({"sentence": sentence, "label": label}, ensure_ascii=False) + "\n")

			with open(f"{NEW_FOLDER_PATH}/test_{i}.json", "w") as f:
				for sentence, label in zip(sentences_test_combined[i], labels_test_combined[i]):
					f.write(json.dumps({"sentence": sentence, "label": label}, ensure_ascii=False) + "\n")

		# merge all categories
		# sentences_train_all = []
		# labels_train_all = []
		# sentences_dev_all = []
		# labels_dev_all = []
		# sentences_test_all = []
		# labels_test_all = []
		# for i in range(1, 8):
		# 	sentences_train_all.extend(sentences_train_combined[i])
		# 	labels_train_all.extend(labels_train_combined[i])
		# 	sentences_dev_all.extend(sentences_dev_combined[i])
		# 	labels_dev_all.extend(labels_dev_combined[i])
		# 	sentences_test_all.extend(sentences_test_combined[i])
		# 	labels_test_all.extend(labels_test_combined[i])


	return sentences_train_combined, labels_train_combined, sentences_dev_combined, labels_dev_combined, sentences_test_combined, labels_test_combined

def main(args):
	process_data_pipeline(args.data_folder_path, args.generate_formatted_data)

if __name__ == "__main__":

	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_folder_path", type=str, default="data")
	parser.add_argument("--generate_formatted_data", action="store_true")

	args = parser.parse_args()


	main(args)
