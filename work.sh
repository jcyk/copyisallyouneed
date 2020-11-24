set -e

dataset=/apdcephfs/private_jcykcai/multi_domain
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.vanilla/epoch80_batch77999_devbleu36.48_testbleu37.34

for domain in it koran law medical subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${domain}/test.txt \
        --output_path ${dataset}/${domain}/test.out.txt
done

