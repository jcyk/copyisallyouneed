set -e
dataset=/apdcephfs/private_jcykcai/esen/full
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.exp.pretrain
for ckpt_folder in exp5/epoch19_batch99999_acc0.99
do
python3 /apdcephfs/private_jcykcai/copyisallyouneed/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${ckpt_prefix}/${ckpt_folder} \
        --dev_batch_size 2048
done

