MODEL_TYPE=$1
DATA_TYPE=$2

list="01 02 03 04 05 06 07 08 09 10"
#list="02 03 04 05"
#list="06 07 08 09 10"

for var in $list
do
    rm -rf "$DATA_TYPE""_$MODEL_TYPE""-base-uncased_train_masked_attention_10.pth"
    rm -rf "$DATA_TYPE""_$MODEL_TYPE""-base-uncased.model"    

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
    --epochs 6
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
    --save_path="result/$MODEL_TYPE""/$var"
    "
    echo $cmd
    eval $cmd
done





