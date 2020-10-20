set -e

ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.3.5.old/epoch32_batch99999_acc0.98
dataset=/apdcephfs/private_jcykcai/esen/3.5
outdir=${ckpt_folder}_3.5
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
cp -r ${ckpt_folder} ${outdir}


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.3.5.old/epoch32_batch99999_acc0.98
dataset=/apdcephfs/private_jcykcai/esen/4.5
outdir=${ckpt_folder}_4.5
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
cp -r ${ckpt_folder} ${outdir}


ckpt_folder=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.3.5.old/epoch32_batch99999_acc0.98
dataset=/apdcephfs/private_jcykcai/esen/full
outdir=${ckpt_folder}_full
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
cp -r ${ckpt_folder} ${outdir}
