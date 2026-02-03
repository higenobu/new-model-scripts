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