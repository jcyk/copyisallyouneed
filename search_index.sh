dataset=/apdcephfs/private_jcykcai/esen
main_folder=${dataset}/ckpt.pretrain
ckpt_folder=${main_folder}/epoch5_batch29999_acc1.00
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/train.src.txt \
        --output_file ${dataset}/train.src.txt.mem \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/src.vocab \
        --index_file ${dataset}/train.tgt.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192
