MODEL_TYPE=$1
DATA_TYPE=$2

list="01 02 03 04 05 06 07 08 09 10"
#list="01 02 03 04 05"


cmd="python train.py \
--dataset $DATA_TYPE \
--split_ratio 0.1 \
--seed 1234  \
--train_type base \
--backbone $MODEL_TYPE \
--classifier_type softmax \
--optimizer adam_vanilla \
--epochs 3
"

echo $cmd
eval $cmd

for var in $list
do
    cmd="python train.py \
    --dataset $DATA_TYPE \
    --split_ratio 0.1 \
    --seed 1234    \
    --train_type masker     \
    --backbone $MODEL_TYPE\
    --classifier_type sigmoid \
    --optimizer adam_masker    \
    --keyword_type attention \
    --lambda_ssl 0.001 \
    --lambda_ent 0.001 \
    --attn_model_path "$DATA_TYPE""_$MODEL_TYPE""-base-uncased.model" \
    --epochs 7
    "
    echo $cmd
    eval $cmd

    cmd="python infer.py --dataset $DATA_TYPE \
    --split_ratio 0.1 \
    --seed 1234 \
    --eval_type ood \
    --ood_datasets remain \
    --backbone $MODEL_TYPE \
    --classifier_type softmax \
    --model_path "$DATA_TYPE""_$MODEL_TYPE""-base-uncased_masker.model" \
    --save_path=result/$var
    "
    echo $cmd
    eval $cmd
done





