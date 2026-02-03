# ktl-ta-scripts
kyototextlab/ktl-psych-press-assist-ds-updated-data における Jupyter Notebook や データセットをまとめ、学習スクリプトに直したものです。

## TA モデル学習

``run.sh`` を実行。
必要な学習パラメーターを適宜変更

```
TARGET_DIR='/home/alkalinemoe/psych_model_scripts/model'

# Model parameters
MAX_LEN=256
BATCH_SIZE=32
LEARNING_RATE=2.9051435624508314e-06
EPOCHS=4
MODEL_NAME_PATH='nlp-waseda/roberta-base-japanese'

# new data path
DATA_PATH='/home/alkalinemoe/psych_model_scripts/data/new/news10000_final_data'


python train.py \
	--data_path $DATA_PATH \
	--max_len $MAX_LEN \
	--batch_size $BATCH_SIZE \
	--learning_rate $LEARNING_RATE \
	--epochs $EPOCHS \
	--output_dir $TARGET_DIR \
	--model_name_or_path $MODEL_NAME_PATH \
```

## 学習済みモデルの推論

``predict.sh`` によりバッチごと推論 (predict) 可能。

```
TARGET_DIR='/home/matsuo/test-output'

# Model parameters
BATCH_SIZE=32
MODEL_NAME_PATH='/home/matsuo/tamodels'

# test data path (.txt file with news text in each line)
TEST_DATA_PATH='/home/alkalinemoe/psych_model_scripts/data/test.txt'

python3 batch_predict.py \
	--data_path $TEST_DATA_PATH \
	--model_name_or_path $MODEL_NAME_PATH \
	--output_dir $TARGET_DIR \
	--batch_size $BATCH_SIZE \
```

動作確認兼ねて学習済みモデルや学習データセットをまとめました。詳細は以下の通りです。

サーバ：`karte-sum-5-image-2` (GC)

データ (学習用再構築)：``/home/alkalinemoe/psych_model_scripts/data``\
学習済みモデル (早大RoBERTa/regression): ``/home/alkalinemoe/psych_model_scripts/model``

```
(番号 1~7 それぞれの感情的情報に対応)
1. 喜びを感じた
2. 恐怖を感じた
3. 驚きを感じた
4. 信頼できる情報と感じた
5. 曖昧な情報と感じた
6. 何かの意図を持って書かれたと感じた
7. 経済に期待がもてると感じた
```
   
学習スクリプト：`/home/alkalinemoe/psych_model_scripts/run.sh`\
推論スクリプト：`/home/alkalinemoe/psych_model_scripts/predict.sh`

(データ・モデル等の指定あり、参考まで)

---

## 評価結果 (+学習ログ)
`/home/alkalinemoe/psych_model_scripts/model/train_logs.txt`

```
Start evaluation

--------------------------------------------------
Evaluating for Title: 喜びを感じた
1 rmse:0.4652856728686356
1 mae:0.3694635147127571
1 correlation(pearson): 0.689108432025275
1 correlation(spearman): 0.6519201154000333
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 恐怖を感じた
2 rmse:0.5347934047399284
2 mae:0.4132513874572481
2 correlation(pearson): 0.6700143230134612
2 correlation(spearman): 0.6093585183149043
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 驚きを感じた
3 rmse:0.518518393828698
3 mae:0.4093776286217721
3 correlation(pearson): 0.4792286946912313
3 correlation(spearman): 0.40989961046426665
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 信頼できる情報と感じた
4 rmse:0.4131677678033597
4 mae:0.32256343025117895
4 correlation(pearson): 0.45266007705170314
4 correlation(spearman): 0.45027666193964705
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 曖昧な情報と感じた
5 rmse:0.45220274709411357
5 mae:0.3495647837297328
5 correlation(pearson): 0.4396915354835116
5 correlation(spearman): 0.44788347797837025
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 何かの意図をもって書かれたと感じた
6 rmse:0.5126121607421786
6 mae:0.4047023584706938
6 correlation(pearson): 0.2973361738616466
6 correlation(spearman): 0.3032301033829848
--------------------------------------------------

--------------------------------------------------
Evaluating for Title: 経済に期待がもてると感じた
7 rmse:0.4842991556509933
7 mae:0.3870875270089656
7 correlation(pearson): 0.7038676308199988
7 correlation(spearman): 0.6479416507413722
--------------------------------------------------
```

## 推論結果 (テストデータ、一部)：
``/home/alkalinemoe/psych_model_scripts/output/preds_batch.txt``

```
--------------------------------------------------
東京2020大会から2周年にあたり、日本オリンピック委員会（JOC）及び日本パラリンピック委員会（JPC）と共同で様々な取り組みを実施します。大会のボランティアや競技団体、区市町村等、多様な主体との連携を深め、大会レガシーの着実な継承・発展を図ってまいります。,
--------------------------------------------------
喜びを感じた: 3.3251454830169678
恐怖を感じた: 1.8006926774978638
驚きを感じた: 2.3238167762756348
信頼できる情報と感じた: 3.5336356163024902
曖昧な情報と感じた: 2.49424147605896
何かの意図をもって書かれたと感じた: 2.785068988800049
経済に期待がもてると感じた: 3.333113431930542


--------------------------------------------------
エンゼルスの大谷翔平投手が「3番・DH」で先発出場。1点を追う4回の第2打席に3試合ぶりとなる26号同点ソロを放った。6月では11本目となる一発。前日に2打席連発の21号、さらに第1打席で22号を放ち、リーグ本塁打王ランキングで大谷に3本差と迫ったロバートJr.（ホワイトソックス）との直接対決で、アーチ競演。その差を再び4本差と広げ、両リーグトップと同時に4試合連続安打を記録。,
--------------------------------------------------
喜びを感じた: 3.093200206756592
恐怖を感じた: 1.72527015209198
驚きを感じた: 2.4806904792785645
信頼できる情報と感じた: 3.8032853603363037
曖昧な情報と感じた: 2.023186683654785
何かの意図をもって書かれたと感じた: 2.571716547012329
経済に期待がもてると感じた: 2.409991502761841
```


## 継続学習用に処理済みデータの作成
``data_utils.py`` によってデータの前処理を実施。

もしカテゴリー別の
``{"sentence":TXT,"label": 数字}`` json であれば、現在のスクリプト (train.py, utils.py, data_utils.py) にパスを指定するだけで TA モデルの学習が可能です。

一方、既存の csv 形式の分割データであれば、data_utils.py の --generate_formatted_data を用い、処理済みの json データの作成が必要となります。\
```
python data_utils.py --data_folder_path /home/alkalinemoe/psych_model_scripts/data/new/news10000_final_data --generate_formatted_data
```
生データの data_folder_path の指定することで data_folder_path ディレクトリに新しい (カテゴリー別の) json が作成されます。\
処理済みの json が作成された後、上記の既存スクリプトが従来通り使えます。
