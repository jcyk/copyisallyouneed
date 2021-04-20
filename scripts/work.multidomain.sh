set -e

### Multidomain testing###


dataset=${MTPATH}/multi_domain
# ckpt=${MTPATH}/mt.ckpts/multi_domain/ckpt.vanilla/best.pt

# for domain in medical law it koran subtitles; do
#     python3 work.py --load_path ${ckpt} \
#         --test_data ${dataset}/${domain}/test.txt \
#         --comp_bleu \
#         --output_path ${dataset}/vanilla.${domain}.test.out.txt
# done



ckpt=${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.dynamic/best.pt

# for domain in medical law it koran subtitles; do
#     python3 work.py --load_path ${ckpt} \
#         --test_data ${dataset}/${domain}/test.txt \
#         --output_path ${dataset}/dynamic.${domain}.test.out.txt \
#         --comp_bleu
# done

# for domain in medical law it koran subtitles; do
#     python3 work.py --load_path ${ckpt} \
#         --index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_${domain} \
#         --test_data ${dataset}/${domain}/test.txt \
#         --output_path ${dataset}/dynamic.domain.${domain}.test.out.txt \
#         --comp_bleu
# done

for domain in medical law it koran subtitles; do
    python3 work.py --load_path ${ckpt} \
        --src_vocab_path ${dataset}/train/src.vocab \
        --tgt_vocab_path ${dataset}/train/tgt.vocab \
        --index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full \
        --test_data ${dataset}/${domain}/test.txt \
        --dump_path ${dataset}/dynamic.full.${domain}.test.out.json \
        --comp_bleu
done
