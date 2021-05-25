set -e


dataset=${MTPATH}/wmt14_gl
index_prefix=${MTPATH}/mt.ckpts/wmt14/ckpt.exp.pretrain

ckpt=${MTPATH}/mt.ckpts/wmt14/ckpt.exp.dynamic/best.pt
index_path=${index_prefix}/epoch3_batch99999_acc0.97
CUDA_VISIBLE_DEVICES=0 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ori.ori.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_ext1
CUDA_VISIBLE_DEVICES=1 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ori.ext.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_all
CUDA_VISIBLE_DEVICES=0 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ori.all.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_fullsearch
CUDA_VISIBLE_DEVICES=1 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ori.full.json \
        --comp_bleu

ckpt=${MTPATH}/mt.ckpts/wmt14/ckpt.exp.dynamic.ext1/best.pt
index_path=${index_prefix}/epoch3_batch99999_acc0.97
CUDA_VISIBLE_DEVICES=0 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ext.ori.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_ext1
CUDA_VISIBLE_DEVICES=1 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ext.ext.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_all
CUDA_VISIBLE_DEVICES=0 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ext.all.json \
        --comp_bleu

index_path=${index_prefix}/epoch3_batch99999_acc0.97_fullsearch
CUDA_VISIBLE_DEVICES=1 python3 work.py --load_path ${ckpt} \
        --index_path ${index_path} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt \
        --dump_path ${dataset}/test.out.ext.full.json \
        --comp_bleu

