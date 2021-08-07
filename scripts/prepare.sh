main=${MTPATH}/gu
out=${MTPATH}

mkdir -p ${out}/enes 
python3 prepare.py --train_data_src ${main}/esen/train/jrc.train.tgt.bpe \
--train_data_tgt ${main}/esen/train/jrc.train.src.bpe \
--vocab_src ${out}/enes/src.vocab \
--vocab_tgt ${out}/enes/tgt.vocab \
--output_file ${out}/enes/train.txt
paste -d '\t' ${main}/esen/dev/jrc.dev.tgt ${main}/esen/dev/jrc.dev.src > ${out}/enes/dev.txt
paste -d '\t' ${main}/esen/test/jrc.test.tgt ${main}/esen/test/jrc.test.src > ${out}/enes/test.txt

mkdir -p ${out}/ende
python3 prepare.py --train_data_src ${main}/deen/train/jrc.train.tgt.bpe \
--train_data_tgt ${main}/deen/train/jrc.train.src.bpe \
--vocab_src ${out}/ende/src.vocab \
--vocab_tgt ${out}/ende/tgt.vocab \
--output_file ${out}/ende/train.txt
paste -d '\t' ${main}/deen/dev/jrc.dev.tgt ${main}/deen/dev/jrc.dev.src > ${out}/ende/dev.txt
paste -d '\t' ${main}/deen/test/jrc.test.tgt ${main}/deen/test/jrc.test.src > ${out}/ende/test.txt

mkdir -p ${out}/esen
python3 prepare.py --train_data_src ${main}/esen/train/jrc.train.src.bpe \
--train_data_tgt ${main}/esen/train/jrc.train.tgt.bpe \
--vocab_src ${out}/esen/src.vocab \
--vocab_tgt ${out}/esen/tgt.vocab \
--output_file ${out}/esen/train.txt
paste -d '\t' ${main}/esen/dev/jrc.dev.src ${main}/esen/dev/jrc.dev.tgt > ${out}/esen/dev.txt
paste -d '\t' ${main}/esen/test/jrc.test.src ${main}/esen/test/jrc.test.tgt > ${out}/esen/test.txt

mkdir -p ${out}/deen
python3 prepare.py --train_data_src ${main}/deen/train/jrc.train.src.bpe \
--train_data_tgt ${main}/deen/train/jrc.train.tgt.bpe \
--vocab_src ${out}/deen/src.vocab \
--vocab_tgt ${out}/deen/tgt.vocab \
--output_file ${out}/deen/train.txt
paste -d '\t' ${main}/deen/dev/jrc.dev.src ${main}/deen/dev/jrc.dev.tgt > ${out}/deen/dev.txt
paste -d '\t' ${main}/deen/test/jrc.test.src ${main}/deen/test/jrc.test.tgt > ${out}/deen/test.txt
