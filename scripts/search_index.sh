set -e

dataset=pdcephfs/share_916081/jcykcai/ende
vocab=${dataset}/src.vocab
ckpt_folder=${MTPATH}/mt.ckpts/ende/ckpt.exp.pretrain/epoch19_batch99999_acc0.97
/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/test.txt \
        --output_file ${dataset}/test.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192



dataset=pdcephfs/share_916081/jcykcai/enes
vocab=${dataset}/src.vocab
ckpt_folder=${MTPATH}/mt.ckpts/enes/ckpt.exp.pretrain/epoch19_batch99999_acc0.99
/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/test.txt \
        --output_file ${dataset}/test.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

exit 0

dataset=pdcephfs/share_916081/jcykcai/esen/full
vocab=${dataset}/src.vocab
ckpt_folder=${MTPATH}/mt.ckpts/esen/ckpt.exp.pretrain4/epoch19_batch99999_acc0.99
/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/test.txt \
        --output_file ${dataset}/test.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192



dataset=pdcephfs/share_916081/jcykcai/deen
vocab=${dataset}/src.vocab
ckpt_folder=${MTPATH}/mt.ckpts/deen/ckpt.exp.pretrain4/epoch19_batch99999_acc0.97
/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/test.txt \
        --output_file ${dataset}/test.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192



dataset=pdcephfs/share_916081/jcykcai/fren
vocab=${dataset}/src.vocab
ckpt_folder=${MTPATH}/mt.ckpts/fren/ckpt.exp.pretrain4/epoch18_batch99999_acc0.99
/search_index.py \
        --input_file ${dataset}/dev.txt \
        --output_file ${dataset}/dev.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/test.txt \
        --output_file ${dataset}/test.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 8192

/search_index.py \
        --input_file ${dataset}/train.txt \
        --output_file ${dataset}/train.mem.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --batch_size 8192

