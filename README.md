# translate
A traning translate project

https://github.com/ictnlp/MoE-Waitk.git

https://github.com/ictnlp/MoE-Waitk

git clone https://github.com/ictnlp/GMA.git

https://github.com/ictnlp/GMA


# Requirements and Installation

    Python version = 3.6

    PyTorch version = 1.7

    Install fairseq:
    cd GMA
    pip install --editable ./
 
# Data Pre-processing:
 bash 1_data_prepare_paper.sh   ##准备数据
 
    src=de
    tgt=en
    train_data="~/wmt15_de_en/train"
    vaild_data="~/wmt15_de_en/valid"
    test_data="~/wmt15_de_en/test"
    data="~/hang/data"

# add --joined-dictionary for WMT15 German-English
    fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --joined-dictionary\     #没有这一行 下一步训练会失败， 加一个字典 使得两个维度一样
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
    
# Training
Train the GMA with the following command:

delta is the relaxation offset to provide a controllable trade-off between translation quality and latency in practice, and we suggest set delta=1.0.
    
    export CUDA_VISIBLE_DEVICES=0  #显卡有几张，= 0 一张显卡
     delta=1.0
     modelfile=~/model1
     data=~/data

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
 
 # Inference
Evaluate the model with the following command:
    export CUDA_VISIBLE_DEVICES=0
    data=~/data
    modelfile=~/hang/model
    ref_dir=~/wmt15_de_en/test.en

    #average last 5 checkpoints
    python scripts/average_checkpoints.py --inputs ${modelfile} --num-epcho-checkpoints 5 --output ${modelfile}/average-model.pt

    #generate translation
    python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    perl multi-bleu.perl -lc ${ref_dir} < pred.translation



# Traing 2 MoE Wait-k
    cd MoE-Waitk
    pip install --editable ./

Train MoE Wait-k Policy in two stage, according to the following command:

    For Transformer-Small with 4 attention heads: we set expert lagging = 1,6,11,16
    For Transformer-Base with 8 attention heads: we set expert lagging = 1,3,5,7,9,11,13,15
    For Transformer-Big with 16 attention heads: we set expert lagging = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
1.First-stage: fix the expert weights equal, and pre-train expert parameters.    

    export CUDA_VISIBLE_DEVICES=0
    data=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/data
    modelfile=/media/hang/1ba83754-8a9c-4989-9cf0-48bf763358da/hang/model1
    #Fisrt-stage: Pertrain an equal-weight MoE Wait-k

    python train.py  --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
     --optimizer adam \
     --adam-betas '(0.9, 0.98)' \
     --clip-norm 0.0 \
     --lr 5e-4 \
     --lr-scheduler inverse_sqrt \
     --warmup-init-lr 1e-07 \
     --warmup-updates 4000 \
     --dropout 0.3 \
     --criterion label_smoothed_cross_entropy \
     --reset-dataloader --reset-lr-scheduler --reset-optimizer\
     --label-smoothing 0.1 \
     --encoder-attention-heads 8 \
     --decoder-attention-heads 8 \
     --left-pad-source False \
     --fp16 \
     --equal-weight \
     --expert-lagging ${expert_lagging} \
     --save-dir ${modelfile} \
     --max-tokens 4096 --update-freq 2
2.Second-stage: jointly ﬁne-tune the parameters of experts and their weights.
    #Sencond-stage: Finetune MoE Wait-k with various expert weights
    
    python train.py  --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
     --optimizer adam \
     --adam-betas '(0.9, 0.98)' \
     --clip-norm 0.0 \
     --lr 5e-4 \
     --lr-scheduler inverse_sqrt \
     --warmup-init-lr 1e-07 \
     --warmup-updates 4000 \
     --dropout 0.3 \
     --criterion label_smoothed_cross_entropy \
     --reset-dataloader --reset-lr-scheduler --reset-optimizer\
     --label-smoothing 0.1 \
     --encoder-attention-heads 8 \
     --decoder-attention-heads 8 \
     --left-pad-source False \
     --fp16 \
     --expert-lagging ${expert_lagging} \
     --save-dir ${modelfile} \
     --max-tokens 4096 --update-freq 2
 
# Inference

Evaluate the model with the following command:
    export CUDA_VISIBLE_DEVICES=0
    data=~/data
    modelfile=~/model1
    ref_dir=~/test.en
    testk=3

    python scripts/average_checkpoints.py --inputs ${modelfile} --num-epoch-checkpoints 5 --output ${modelfile}/average-model.pt 

    #generate translation
    python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} > pred.out

    grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
    perl multi-bleu.perl -lc ${ref_dir} < pred.translation


1.数据下载 bash 1_data_prepare_paper.sh

2.环境2 问题 conda env create -f moewaitk.yaml 

2.1  切换到  conda activate torch17 然后 pip install -r 17.txt

