set -e



dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.full.old/epoch19_batch99999_acc0.98 \
        --dev_batch_size 2048



dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/retriever/ckpt.pretrain.neg.2/epoch9_batch99999_acc0.02 \
        --dev_batch_size 2048


dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/retriever/ckpt.pretrain.neg.0/epoch9_batch99999_acc0.02 \
        --dev_batch_size 2048


dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/test_retriever.py \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/retriever/ckpt.pretrain.neg.33/epoch9_batch99999_acc0.02 \
        --dev_batch_size 2048
