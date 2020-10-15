index_dataset=/apdcephfs/private_jcykcai/esen/full
dataset=/apdcephfs/private_jcykcai/esen/1.5
ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.1.5/epoch97_batch99999_acc0.93
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/dev.src.txt \
        --output_file ${dataset}/dev.src.txt.mem \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${index_dataset}/src.vocab \
        --index_file ${index_dataset}/train.tgt.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

dataset=/apdcephfs/private_jcykcai/esen/2.5
ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.2.5/epoch48_batch99999_acc0.95
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/dev.src.txt \
        --output_file ${dataset}/dev.src.txt.mem \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${index_dataset}/src.vocab \
        --index_file ${index_dataset}/train.tgt.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

dataset=/apdcephfs/private_jcykcai/esen/3.5
ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.3.5/epoch32_batch99999_acc0.95
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/dev.src.txt \
        --output_file ${dataset}/dev.src.txt.mem \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${index_dataset}/src.vocab \
        --index_file ${index_dataset}/train.tgt.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

dataset=/apdcephfs/private_jcykcai/esen/4.5
ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.4.5/epoch24_batch99999_acc0.95
python3 /apdcephfs/private_jcykcai/copyisallyouneed/search_index.py \
        --input_file ${dataset}/dev.src.txt \
        --output_file ${dataset}/dev.src.txt.mem \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${index_dataset}/src.vocab \
        --index_file ${index_dataset}/train.tgt.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

