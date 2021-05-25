set -e
dataset=${MTPATH}/esen/full
ckpt_prefix=${MTPATH}/mt.ckpts
for ckpt_folder in esen/ckpt.exp.pretrain3/epoch19_batch99999_acc0.97 esen/ckpt.exp.pretrain4/epoch19_batch99999_acc0.99
do
/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${ckpt_prefix}/${ckpt_folder} \
        --dev_batch_size 2048
done

dataset=${MTPATH}/deen
ckpt_prefix=${MTPATH}/mt.ckpts
for ckpt_folder in deen/ckpt.exp.pretrain3/epoch19_batch99999_acc0.95 deen/ckpt.exp.pretrain4/epoch19_batch99999_acc0.97
do
/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${ckpt_prefix}/${ckpt_folder} \
        --dev_batch_size 2048
done

dataset=${MTPATH}/fren
ckpt_prefix=${MTPATH}/mt.ckpts
for ckpt_folder in fren/ckpt.exp.pretrain3/epoch18_batch99999_acc0.99 fren/ckpt.exp.pretrain4/epoch18_batch99999_acc0.99
do
/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${ckpt_prefix}/${ckpt_folder} \
        --dev_batch_size 2048
done

