set -e

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.neg.2/epoch9_batch99999_acc0.02
dataset=/apdcephfs/private_jcykcai/esen/full
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

cp ${dataset}/train.tgt.txt ${ckpt_folder}/candidates.txt


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.neg.0/epoch9_batch99999_acc0.02
dataset=/apdcephfs/private_jcykcai/esen/full
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

cp ${dataset}/train.tgt.txt ${ckpt_folder}/candidates.txt


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.neg.33/epoch9_batch99999_acc0.02
dataset=/apdcephfs/private_jcykcai/esen/full
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

cp ${dataset}/train.tgt.txt ${ckpt_folder}/candidates.txt


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.rnd.2/epoch19_batch99999_acc0.98
dataset=/apdcephfs/private_jcykcai/esen/full
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

cp ${dataset}/train.tgt.txt ${ckpt_folder}/candidates.txt


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.rnd.0/epoch19_batch99999_acc0.98
dataset=/apdcephfs/private_jcykcai/esen/full
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

cp ${dataset}/train.tgt.txt ${ckpt_folder}/candidates.txt
