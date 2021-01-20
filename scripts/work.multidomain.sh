set -e

### Multidomain testing###


dataset=/apdcephfs/private_jcykcai/multi_domain
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.vanilla/best.pt

for domain in medical law it koran subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${domain}/test.txt \
        --comp_bleu \
        --output_path ${dataset}/vanilla.${domain}.test.out.txt
done



ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.dynamic/best.pt

for domain in medical law it koran subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${domain}/test.txt \
        --output_path ${dataset}/dynamic.${domain}.test.out.txt \
        --comp_bleu
done

for domain in medical law it koran subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --index_path /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_${domain} \
        --test_data ${dataset}/${domain}/test.txt \
        --output_path ${dataset}/dynamic.domain.${domain}.test.out.txt \
        --comp_bleu
done

for domain in medical law it koran subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --index_path /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full \
        --test_data ${dataset}/${domain}/test.txt \
        --output_path ${dataset}/dynamic.full.${domain}.test.out.txt \
        --comp_bleu
done
