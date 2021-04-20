set -e

ckpt_prefix=${MTPATH}/mt.ckpts
dataset_prefix=pdcephfs/share_916081/jcykcai/esen

ckpt_folder=esen/ckpt.exp.pretrain1.4/epoch78_batch99999_acc0.99
for dataset in 1.4 2.4 3.4 full
do
echo ${ckpt_prefix}/${ckpt_folder} ${dataset_prefix}/${dataset}
/build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset_prefix}/${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${ckpt_prefix}/${ckpt_folder}_${dataset}
done

ckpt_folder=esen/ckpt.exp.pretrain2.4/epoch39_batch99999_acc0.99
for dataset in 2.4 3.4 full
do
echo ${ckpt_prefix}/${ckpt_folder} ${dataset_prefix}/${dataset}
/build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset_prefix}/${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${ckpt_prefix}/${ckpt_folder}_${dataset}
done

