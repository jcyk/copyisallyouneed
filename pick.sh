dataset=/apdcephfs/private_jcykcai/esen
main_folder=${dataset}/ckpt.pretrain.6layers
ckpt_folder=${main_folder}/epoch19_batch99999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/pick_shared_encoder.py \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --input_path ${ckpt_folder} \
        --output_path ${ckpt_folder}/shared_encoder
