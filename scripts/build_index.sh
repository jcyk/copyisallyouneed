set -e

ckpt_prefix=${MTPATH}/mt.ckpts

dataset=pdcephfs/share_916081/jcykcai/wmt14_gl
ckpt_folder=wmt14/ckpt.exp.pretrain.gl/epoch3_batch99999_acc0.97

output_folder=${ckpt_prefix}/${ckpt_folder}
/build_index.py \
        --input_file ${dataset}/cands.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset}/cands.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/cands.txt ${output_folder}/candidates.txt

exit 0

dataset=pdcephfs/share_916081/jcykcai/multi_domain
ckpt_folder=multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80

output_folder=${ckpt_prefix}/${ckpt_folder}_gs
/build_index.py \
        --input_file ${dataset}/gs/gs.cands.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/train/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset}/gs/gs.cands.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/train/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat
rm -rf $output_folder
cp ${dataset}/gs/gs.cands.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${output_folder}


ckpt=${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.dynamic/best.pt

python3 work.py --load_path ${ckpt} \
        --index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full \
        --test_data ${dataset}/gs/gs.test.txt \
        --comp_bleu

python3 work.py --load_path ${ckpt} \
        --index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_gs \
        --test_data ${dataset}/gs/gs.test.txt \
        --comp_bleu
exit 0







for domain in train it koran law medical subtitles full; do
output_folder=${ckpt_prefix}/${ckpt_folder}_${domain}
/build_index.py \
        --input_file ${dataset}/${domain}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/train/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset}/${domain}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/train/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat
rm -rf $output_folder
cp ${dataset}/${domain}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${output_folder}
done

exit 0

dataset=pdcephfs/share_916081/jcykcai/ende
ckpt_folder=ende/ckpt.exp.pretrain/epoch19_batch99999_acc0.97 
echo ${ckpt_prefix}/${ckpt_folder}
/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt


dataset=pdcephfs/share_916081/jcykcai/enes
ckpt_folder=enes/ckpt.exp.pretrain/epoch19_batch99999_acc0.99
echo ${ckpt_prefix}/${ckpt_folder}
/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192

/build_index.py \
        --input_file ${dataset}/train.tgt.txt \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset}/tgt.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --batch_size 8192 \
        --only_dump_feat

cp ${dataset}/train.tgt.txt ${ckpt_prefix}/${ckpt_folder}/candidates.txt
