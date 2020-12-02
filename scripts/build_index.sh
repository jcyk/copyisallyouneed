set -e

ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts


dataset=/apdcephfs/private_jcykcai/multi_domain
ckpt_folder=multi_domain/2.4/ckpt.exp.pretrain/epoch20_batch99999_acc0.81
for domain in it koran law medical subtitles full 2.4; do
output_folder=${ckpt_prefix}/${ckpt_folder}_${domain}
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/${domain}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/2.4/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/${domain}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/2.4/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat
rm -rf $output_folder
cp ${dataset}/${domain}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${output_folder}
done

exit 0

dataset=/apdcephfs/private_jcykcai/ende
ckpt_folder=ende/ckpt.exp.pretrain/epoch19_batch99999_acc0.97 
echo ${ckpt_prefix}/${ckpt_folder}
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt


dataset=/apdcephfs/private_jcykcai/enes
ckpt_folder=enes/ckpt.exp.pretrain/epoch19_batch99999_acc0.99
echo ${ckpt_prefix}/${ckpt_folder}
python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

python3 /apdcephfs/private_jcykcai/copyisallyouneed/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
