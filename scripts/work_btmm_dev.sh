set -e

for dir in esen enes deen ende; do
    ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/${dir}/transfer/btmm1/best.pt
    dataset=/apdcephfs/private_jcykcai/${dir}
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.txt \
       --comp_bleu
done

for dir in esen enes deen ende; do
    ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/${dir}/transfer/btmm2/best.pt
    dataset=/apdcephfs/private_jcykcai/${dir}
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.txt \
       --comp_bleu
done
