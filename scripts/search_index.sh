vocab=/apdcephfs/private_jcykcai/esen/full/src.vocab
dataset=/apdcephfs/private_jcykcai/esen/full
ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.exp.pretrain/exp5/epoch19_batch99999_acc0.99
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192


