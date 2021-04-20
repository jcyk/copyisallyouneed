set -e

for dir in esen enes deen ende; do
    ckpt=${MTPATH}/mt.ckpts/${dir}/transfer/btmm1/best.pt
    ${MTPATH}/${dir}
    python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.txt \
       --comp_bleu
done

for dir in esen enes deen ende; do
    ckpt=${MTPATH}/mt.ckpts/${dir}/transfer/btmm2/best.pt
    ${MTPATH}/${dir}
    python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.txt \
       --comp_bleu
done
