set -e

dataset=/apdcephfs/private_jcykcai/esen
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen
    
for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla/epoch32_batch98999_devbleu63.81_testbleu64.26
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.bm25/epoch25_batch75999_devbleu66.10_testbleu65.79
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.bm25.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.static/epoch28_batch86999_devbleu66.14_testbleu66.09
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.mem.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.exp.dynamic/epoch21_batch64999_devbleu66.81_testbleu66.49
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done



for split in dev test; do
    ckpt=${ckpt_prefix}/ckpt.vanilla.1.4/epoch128_batch96999_devbleu58.77_testbleu58.36
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to1/epoch125_batch94999_devbleu58.49_testbleu58.19
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to2/epoch23_batch17999_devbleu59.49_testbleu59.18
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to3/epoch23_batch17999_devbleu60.05_testbleu60.12
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/1to4/epoch18_batch13999_devbleu60.60_testbleu60.70
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/ckpt.vanilla2.4/epoch60_batch91999_devbleu61.94_testbleu61.66
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to2/epoch48_batch73999_devbleu62.73_testbleu62.37
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to3/epoch46_batch69999_devbleu63.51_testbleu63.29
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt

    ckpt=${ckpt_prefix}/transfer/2to4/epoch20_batch30999_devbleu64.59_testbleu64.09
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work100.py --load_path ${ckpt} \
    --test_data ${dataset}/${split}.txt \
    --output_path ${dataset}/${split}.out.txt
done
