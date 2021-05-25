set -e

#ckpt=${MTPATH}/mt.ckpts/esen/ckpt.bm25/epoch25_batch77999_devbleu66.98_testbleu66.48
#dataset=${MTPATH}/esen
#python3 work.py --load_path ${ckpt} \
#       --test_data ${dataset}/test.bm25.txt \
#       --src_vocab_path ${dataset}/src.vocab \
#       --tgt_vocab_path ${dataset}/tgt.vocab \
#       --output_path ${dataset}/test.out.bm25.txt \
#       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/enes/ckpt.bm25/epoch30_batch92999_devbleu63.04_testbleu62.76
dataset=${MTPATH}/enes
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.bm25.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.bm25.txt  \
       --comp_bleu

exit 0

ckpt=${MTPATH}/mt.ckpts/ende/ckpt.bm25/epoch24_batch74999_devbleu57.88_testbleu57.53
dataset=${MTPATH}/ende
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.bm25.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.bm25.txt  \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/deen/ckpt.bm25/epoch28_batch86999_devbleu63.62_testbleu63.85
dataset=${MTPATH}/deen
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.bm25.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.bm25.txt  \
       --comp_bleu

#--index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full
