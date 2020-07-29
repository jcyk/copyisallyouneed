dataset=/apdcephfs/private_jcykcai/esen
main_folder={dataset}/ckpt.pretrain
ckpt_folder={main_folder}/epoch5_batch27999_acc1.00
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index
