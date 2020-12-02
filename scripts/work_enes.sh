set -e

dataset=/apdcephfs/private_jcykcai/enes
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes
    
for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla/epoch32_batch97999_devbleu61.43_testbleu60.96
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.bm25/epoch28_batch86999_devbleu62.35_testbleu62.33
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.bm25.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.static/epoch24_batch72999_devbleu62.09_testbleu62.13
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.mem.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.dynamic/epoch30_batch91999_devbleu62.85_testbleu63.27
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done


for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla.1.4/epoch102_batch76999_devbleu57.01_testbleu56.27
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to1/epoch38_batch28999_devbleu56.33_testbleu55.94
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to2/epoch22_batch16999_devbleu57.22_testbleu56.56
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to3/epoch23_batch17999_devbleu56.94_testbleu56.90
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to4/epoch19_batch14999_devbleu57.72_testbleu57.21
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.vanilla2.4/epoch48_batch72999_devbleu59.93_testbleu59.32
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to2/epoch39_batch58999_devbleu59.88_testbleu59.75
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to3/epoch31_batch46999_devbleu60.53_testbleu60.65
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to4/epoch51_batch77999_devbleu61.21_testbleu61.01
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done
