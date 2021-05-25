set -e
#${MTPATH}/mt.ckpts/wmt14/ckpt.exp.pretrain/epoch3_batch99999_acc0.97/

ckpt_prefix=${MTPATH}/mt.ckpts

dataset=${MTPATH}/wmt14_gl
ckpt_folder=wmt14/ckpt.exp.pretrain/epoch3_batch99999_acc0.97



#ext=$1
ckpt=${MTPATH}/mt.ckpts/wmt14/ckpt.exp.dynamic.ext1/best.pt

output_folder=${ckpt_prefix}/${ckpt_folder}_fullsearch

rm -rf ${output_folder}
cp -r ${ckpt_prefix}/${ckpt_folder} ${output_folder}

python3 build_index.py \
        --input_file ${dataset}/fullsearch.tgt.txt \
        --ckpt_path  ${output_folder}/response_encoder \
        --args_path ${output_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${output_folder}/mips_index \
        --batch_size 8192

python3 build_index.py \
        --input_file ${dataset}/fullsearch.tgt.txt \
        --ckpt_path ${output_folder}/response_encoder \
        --args_path ${output_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${output_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/fullsearch.tgt.txt ${output_folder}/candidates.txt


python3 work.py --load_path ${ckpt} \
    --index_path ${output_folder} \
    --test_data ${dataset}/test.txt \
    --output_path ${dataset}/test.out.fullsearch.txt \
    --dump_path ${dataset}/test.out.fullsearch.json \
    --comp_bleu


# for ext in a b c d e f g h; do
# output_folder=${ckpt_prefix}/${ckpt_folder}_a${ext}
# python3 work.py --load_path ${ckpt} \
#     --index_path ${output_folder} \
#     --test_data ${dataset}/test.txt \
#     --output_path ${dataset}/test.out.a${ext}.txt \
#     --dump_path ${dataset}/test.out.a${ext}.json \
#     --comp_bleu
# done
