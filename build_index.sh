dataset=/apdcephfs/private_jcykcai/esen
main_folder=${dataset}/ckpt.pretrain.33
ckpt_folder=${main_folder}/epoch16_batch83999_acc0.97
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

ckpt_folder=${main_folder}/epoch16_batch83999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192
