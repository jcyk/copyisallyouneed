set -e

dataset=pdcephfs/share_916081/jcykcai/ende/2.4
reverse_dataset=pdcephfs/share_916081/jcykcai/deen
ckpt=${MTPATH}/mt.ckpts/ende/bt2.4

#awk -F '\t' '{print $2"\t"$1}' ${dataset}/train.txt > ${dataset}/train.reverse.txt

#/train.py --train_data ${dataset}/train.reverse.txt \
#        --dev_data ${reverse_dataset}/dev.txt \
#        --test_data ${reverse_dataset}/test.txt \
#        --src_vocab ${reverse_dataset}/src.vocab \
#        --tgt_vocab ${reverse_dataset}/tgt.vocab \
#        --ckpt ${ckpt} \
#        --world_size 2 \
#        --gpus 2 \
#        --arch vanilla \
#        --dev_batch_size 2048 \
#        --per_gpu_train_batch_size 4096 \
#        --only_save_best

#python3 work.py --load_path ${ckpt}/best.pt \
#        --test_data ${reverse_dataset}/train.txt \
#        --output_path ${dataset}/bt.greedy.train.tgt.txt \
#        --bt \
#        --beam_size 1

#awk -F '\t' '{print $1}' ${reverse_dataset}/train.txt > ${reverse_dataset}/train.src.txt
#rm -f ${dataset}/bt.greedy.train.txt
#paste -d '\t' ${dataset}/bt.greedy.train.tgt.txt ${reverse_dataset}/train.src.txt >> ${dataset}/bt.greedy.train.txt
#cat ${dataset}/train.txt >> ${dataset}/bt.greedy.train.txt

#python3 work.py --load_path ${ckpt}/best.pt \
#        --test_data ${reverse_dataset}/train.txt \
#        --output_path ${dataset}/bt.beam.train.tgt.txt \
#        --bt \
#        --beam_size 5

#rm -f ${dataset}/bt.beam.train.txt
#paste -d '\t' ${dataset}/bt.beam.train.tgt.txt ${reverse_dataset}/train.src.txt >> ${dataset}/bt.beam.train.txt
#cat ${dataset}/train.txt >> ${dataset}/bt.beam.train.txt

#/train.py --train_data ${dataset}/bt.greedy.train.txt \
#        --dev_data ${dataset}/dev.txt \
#        --test_data ${dataset}/test.txt \
#        --src_vocab ${dataset}/src.vocab \
#        --tgt_vocab ${dataset}/tgt.vocab \
#        --ckpt ${ckpt}/greedy \
#        --world_size 2 \
#        --gpus 2 \
#        --arch vanilla \
#        --dev_batch_size 2048 \
#        --per_gpu_train_batch_size 4096

/train.py --train_data ${dataset}/bt.beam.train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${ckpt}/beam \
        --world_size 2 \
        --gpus 2 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096




