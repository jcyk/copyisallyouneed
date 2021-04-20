set -e

ckpt_prefix=${MTPATH}/mt.ckpts
dataset_prefix=${MTPATH}/deen

ckpt_folder=deen/ckpt.exp.pretrain1.4/epoch77_batch99999_acc0.97
for dataset in 1.4 2.4 3.4 full
do
echo ${ckpt_prefix}/${ckpt_folder} ${dataset_prefix}/${dataset}
python3 build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

python3 build_index.py \
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

ckpt_folder=deen/ckpt.exp.pretrain2.4/epoch38_batch99999_acc0.98
for dataset in 2.4 3.4 full
do
echo ${ckpt_prefix}/${ckpt_folder} ${dataset_prefix}/${dataset}
python3 build_index.py \
        --input_file ${dataset_prefix}/${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

python3 build_index.py \
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

