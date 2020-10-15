dataset=/apdcephfs/private_jcykcai/esen/full


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.full.old/epoch19_batch99999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.1.5.old/epoch97_batch99999_acc0.97
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.2.5.old/epoch48_batch99999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.3.5.old/epoch32_batch99999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.4.5.old/epoch19_batch80999_acc0.98
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_folder}/response_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat


