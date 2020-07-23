main=/apdcephfs/share_916081/jcykcai/gu
out=/apdcephfs/private_jcykcai

python3 prepare.py --train_data_src ${main}/esen/train/jrc.train.src.bpe \
--train_data_tgt ${main}/esen/train/jrc.train.tgt.bpe \
--vocab_src ${out}/esen/es.vocab \
--vocab_tgt ${out}/esen/en.vocab \
--output_file ${out}/esen/train.txt
paste -d '\t' ${main}/esen/dev/jrc.dev.src ${main}/esen/dev/jrc.dev.tgt > ${out}/esen/dev.txt
paste -d '\t' ${main}/esen/test/jrc.test.src ${main}/esen/test/jrc.test.tgt > ${out}/esen/test.txt

python3 prepare.py --train_data_src ${main}/fren/train/jrc.train.src.bpe \
--train_data_tgt ${main}/fren/train/jrc.train.tgt.bpe \
--vocab_src ${out}/fren/fr.vocab \
--vocab_tgt ${out}/fren/en.vocab \
--output_file ${out}/fren/train.txt
paste -d '\t' ${main}/fren/dev/jrc.dev.src ${main}/fren/dev/jrc.dev.tgt > ${out}/fren/dev.txt
paste -d '\t' ${main}/fren/test/jrc.test.src ${main}/fren/test/jrc.test.tgt > ${out}/fren/test.txt

python3 prepare.py --train_data_src ${main}/deen/train/jrc.train.src.bpe \
--train_data_tgt ${main}/deen/train/jrc.train.tgt.bpe \
--vocab_src ${out}/deen/de.vocab \
--vocab_tgt ${out}/deen/en.vocab \
--output_file ${out}/deen/train.txt
paste -d '\t' ${main}/deen/dev/jrc.dev.src ${main}/deen/dev/jrc.dev.tgt > ${out}/deen/dev.txt
paste -d '\t' ${main}/deen/test/jrc.test.src ${main}/deen/test/jrc.test.tgt > ${out}/deen/test.txt