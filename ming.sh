src=de
tgt=en
train_data="/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en/train"
vaild_data="/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en/valid"
test_data="/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en/test"
data="/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data"

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --joined-dictionary\
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20

export CUDA_VISIBLE_DEVICES=0
data=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data
modelfile=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/model1
ref_dir=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation


 
export CUDA_VISIBLE_DEVICES=4
 delta=1.0
 modelfile=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/model1
 data=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data

python train.py --ddp-backend=no_c10d ${data} \
 --arch transformer \
 --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --delta ${delta} \
 --save-dir ${modelfile} \
 --max-tokens 4096 --update-freq 2

export CUDA_VISIBLE_DEVICES=0
data=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data
modelfile=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/model1
ref_dir=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en

# average last 5 checkpoints
python3 scripts/average_checkpoints.py --inputs ${modelfile} --num-update-
checkpoints 5 --output ${modelfile}/average-model.pt

# generate translation
python3 generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation






##chenggong right
export CUDA_VISIBLE_DEVICES=0
data=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data
modelfile=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/model1
ref_dir=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/wmt15_de_en/test.en
testk=3
python scripts/average_checkpoints.py --inputs ${modelfile} --num-epoch-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
perl multi-bleu.perl -lc ${ref_dir} < pred.translation
