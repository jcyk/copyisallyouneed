#!/bin/bash
set -e

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPE_TOKENS=32000
BPE_CODE=bpe.code

src=de
tgt=en
tmp=tmp

DOMAINS="it  koran  law  medical  subtitles"
if false; then
    mkdir -p $tmp
    rm -f $tmp/train.tok.all.$src $tmp/train.tok.all.$tgt
    echo "pre-processing train data..."
    for f in $DOMAINS; do
        mkdir $tmp/$f
        for l in $src $tgt; do
            cat $f/train.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $l > $tmp/$f/train.raw.tok.$l
        done
        perl $CLEAN $tmp/$f/train.raw.tok $src $tgt $tmp/$f/train.tok 1 100
        cat $tmp/$f/train.tok.$src >> $tmp/train.tok.all.$src
        cat $tmp/$f/train.tok.$tgt >> $tmp/train.tok.all.$tgt
    done
    subword-nmt learn-joint-bpe-and-vocab --input $tmp/train.tok.all.$src $tmp/train.tok.all.$tgt -s $BPE_TOKENS -o $BPE_CODE --write-vocabulary vocab.$src vocab.$tgt
    
    for f in $DOMAINS; do
        subword-nmt apply-bpe -c $BPE_CODE --vocabulary vocab.$src < $tmp/$f/train.tok.$src > $f/train.src.bpe
        subword-nmt apply-bpe -c $BPE_CODE --vocabulary vocab.$tgt < $tmp/$f/train.tok.$tgt > $f/train.tgt.bpe
    done


echo "pre-processing dev/test data..."
for f in $DOMAINS; do
    for l in $src $tgt; do
        cat $f/dev.$l | perl $NORM_PUNC $l | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l $l > $tmp/$f/dev.tok.$l
        cat $f/test.$l | perl $NORM_PUNC $l | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l $l > $tmp/$f/test.tok.$l
    done
done

for split in dev test; do
    for f in $DOMAINS; do
        subword-nmt apply-bpe -c $BPE_CODE --vocabulary vocab.$src < $tmp/$f/$split.tok.$src > $f/$split.src.bpe
        subword-nmt apply-bpe -c $BPE_CODE --vocabulary vocab.$tgt < $tmp/$f/$split.tok.$tgt > $f/$split.tgt.bpe
    done
done
fi

rm -f dev.txt test.txt
mkdir -p dev test
for f in $DOMAINS; do
    for split in train dev test; do
        if [ "$split" != "train" ]; then
            paste -d '\t' $f/$split.src.bpe $f/$split.tgt.bpe > $f/$split.txt
        else
            paste -d '\t' $f/$split.src.bpe $f/$split.tgt.bpe | shuf > $f/$split.src.tgt.bpe
        fi
    done
    cat $f/dev.txt >> dev.txt
    cat $f/test.txt >> test.txt
    cp $f/dev.txt dev/$f.dev.txt
    cp $f/test.txt test/$f.test.txt 
done


mkdir -p train
rm -f train/train.src.bpe train/train.tgt.bpe
for f in $DOMAINS; do
    awk '{if (NR%4 == 0)  print $0; }' $f/train.src.tgt.bpe > train.src.tgt.bpe
    awk -F '\t' '{print $1}' train.src.tgt.bpe > train/$f.train.src.bpe
    awk -F '\t' '{print $2}' train.src.tgt.bpe > train/$f.train.tgt.bpe
    cat train/$f.train.src.bpe >> train/train.src.bpe
    cat train/$f.train.tgt.bpe >> train/train.tgt.bpe
    rm train.src.tgt.bpe
done

