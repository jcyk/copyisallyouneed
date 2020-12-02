set -e

dataset=/apdcephfs/private_jcykcai/deen
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen

for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla/epoch32_batch98999_devbleu59.40_testbleu59.94
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.bm25/epoch31_batch95999_devbleu63.22_testbleu62.99
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.bm25.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.static/epoch26_batch81999_devbleu62.99_testbleu62.57
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.mem.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.dynamic/epoch32_batch97999_devbleu63.82_testbleu63.42
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done

for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla.1.4/epoch119_batch90999_devbleu54.02_testbleu54.26
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
    
    ckpt=${ckpt_prefix}/transfer/1to1/epoch123_batch93999_devbleu53.76_testbleu53.94
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to2/epoch37_batch28999_devbleu54.67_testbleu54.37
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to3/epoch20_batch15999_devbleu55.97_testbleu55.96
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to4/epoch20_batch15999_devbleu57.04_testbleu56.64
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.vanilla2.4/epoch62_batch94999_devbleu57.58_testbleu58.04
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to2/epoch60_batch92999_devbleu58.91_testbleu58.82
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to3/epoch33_batch51999_devbleu60.05_testbleu59.53
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to4/epoch29_batch44999_devbleu61.32_testbleu60.62
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done
