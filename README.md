# UniLM
**The repository has been cloned from an older commit of the original [UniLM](https://github.com/microsoft/unilm) repository and modified to add some special tokens for training into vocabulary 

## Environment

### Docker

The recommended way to run the code is using docker under Linux:
```bash
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel bash
```

The docker is initialized by:
```bash
. .bashrc
apt-get update
apt-get install -y vim wget ssh

PWD_DIR=$(pwd)
cd $(mktemp -d)
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --user --cuda_ext --cpp_ext
cd $PWD_DIR

pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
python -c "import nltk; nltk.download('punkt')"
pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
```
The mixed-precision training code requires the specific version of [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/1603407bf49c7fc3da74fceb6a6c7b47fece2ef8), which only supports pytorch<1.2.0.

Install the repo as a package in the docker:
```bash
mkdir ~/code; cd ~/code
git clone https://github.com/cb1711/unilm_MetaGen.git
cd ~/code/unilm_MetaGen/src
pip install --user --editable .
```


## Fine-tuning
We provide instructions on how to fine-tune UniLM as a sequence-to-sequence model to support various downstream natural language generation tasks as follows. 

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/data
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 1 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.src --tgt_file train.tgt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 568 --max_position_embeddings 568 \
  --trunc_seg a --always_truncate_tail --max_len_b 120 \
  --mask_prob 0.7 --max_pred 120 \
  --train_batch_size 16 --gradient_accumulation_steps 16 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 80
```
Finetuned checkpoint for evaluation will be uploaded when COVID-19 situation gets back to normal

```bash
# run decoding
DATA_DIR=/{path_of_data}/data
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 568 --max_tgt_length 120 \
  --batch_size 16 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
