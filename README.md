# CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation

This is the official PyTorch implementation for the following EMNLP 2021 paper from Salesforce Research:

**Title**: [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf)

**Authors**: [Yue Wang](https://yuewang-cuhk.github.io/), [Weishi Wang](https://www.linkedin.com/in/weishi-wang/)
, [Shafiq Joty](https://raihanjoty.github.io/), and [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home)

![CodeT5 demo](codet5.gif)

## Updates

**Oct 29, 2021**

We
release [fine-tuned checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/finetuned_models)
for all the downstream tasks covered in the paper.

**Oct 25, 2021**

We release a CodeT5-base fine-tuned
checkpoint ([Salesforce/codet5-base-multi-sum](https://huggingface.co/Salesforce/codet5-base-multi-sum)) for
multilingual code summarzation. Below is how to use this model:

```python
from transformers import RobertaTokenizer, T5ForConditionalGeneration

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    text = """def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG data.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image"""

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=20)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    # this prints: "Convert a SVG string to a QImage."
```

**Oct 18, 2021**

We add a [model card](https://github.com/salesforce/CodeT5/blob/main/CodeT5_model_card.pdf) for CodeT5! Please reach out
if you have any questions about it.

**Sep 24, 2021**

CodeT5 is now in [hugginface](https://huggingface.co/)!

You can simply load the model ([CodeT5-small](https://huggingface.co/Salesforce/codet5-small)
and [CodeT5-base](https://huggingface.co/Salesforce/codet5-base)) and do the inference:

```python
from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=8)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "{user.username}"
```

## Introduction

This repo provides the code for reproducing the experiments
in [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf)
. CodeT5 is a new pre-trained encoder-decoder model for programming languages, which is pre-trained on **8.35M**
functions in 8 programming languages (Python, Java, JavaScript, PHP, Ruby, Go, C, and C#). In total, it achieves
state-of-the-art results on **14 sub-tasks** in a code intelligence benchmark - [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).

Paper link: https://arxiv.org/abs/2109.00859

Blog link: https://blog.salesforceairesearch.com/codet5/

The code currently includes two pre-trained checkpoints ([CodeT5-small](https://huggingface.co/Salesforce/codet5-small)
and [CodeT5-base](https://huggingface.co/Salesforce/codet5-base)) and scripts to fine-tine them on 4 generation tasks (
code summarization, code generation, translation, and refinement) plus 2 understanding tasks (code defect detection and
clone detection) in CodeXGLUE. We also provide their fine-tuned checkpoints to facilitate the easy replication
of our paper.

In practice, CodeT5 can be deployed as an AI-powered coding assistant to boost the productivity of software developers.
At Salesforce, we build an [AI coding assistant demo](https://github.com/salesforce/CodeT5/raw/main/codet5.gif) using
CodeT5 as a VS Code plugin to provide three capabilities for Apex developers:

- **Text-to-code generation**: generate code based on the natural language description.
- **Code autocompletion**: complete the whole function of code given the target function name.
- **Code summarization**: generate the summary of a function in natural language description.

## Table of Contents

1. [Citation](#citation)
2. [License](#license)
3. [Dependency](#dependency)
4. [Download](#download)
5. [Fine-tuning](#fine-tuning)
6. [Get Involved](#get-involved)

## Citation

If you find this code to be useful for your research, please consider citing.

```
@inproceedings{
    wang2021codet5,
    title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation}, 
    author={Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi},
    booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021},
    year={2021},
}
```

## License

The code is released under the BSD-3 License (see `LICENSE.txt` for details), but we also ask that users respect the
following:

This software should not be used to promote or profit from:

violence, hate, and division,

environmental destruction,

abuse of human rights, or

the destruction of people's physical and mental health.

We encourage users of this software to tell us about the applications in which they are putting it to use by emailing
codeT5@salesforce.com, and to
use [appropriate](https://arxiv.org/abs/1810.03993) [documentation](https://www.partnershiponai.org/about-ml/) when
developing high-stakes applications of this model.

## Dependency

- Pytorch 1.7.1
- tensorboard 2.4.1
- transformers 4.6.1
- tree-sitter 0.2.2

## Download

* [Pre-trained checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/pretrained_models)
* [Fine-tuning data](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/data)
* [Fine-tuned checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/finetuned_models)

Instructions to download:

```
# pip install gsutil
cd your-cloned-codet5-path

gsutil -m cp -r "gs://sfr-codet5-data-research/pretrained_models" .
gsutil -m cp -r "gs://sfr-codet5-data-research/data" .
gsutil -m cp -r "gs://sfr-codet5-data-research/finetuned_models" .
```

## Fine-tuning

Go to `sh` folder, set the `WORKDIR` in `exp_with_args.sh` to be your cloned CodeT5 repository path.

You can use `run_exp.py` to run a broad set of experiments by simply passing the `model_tag`, `task`, and `sub_task`
arguments. In total, we support five models (i.e., ['roberta', 'codebert', 'bart_base', 'codet5_small', 'codet5_base'])
and six tasks (i.e., ['summarize', 'concode', 'translate', 'refine', 'defect', 'clone']). For each task, we use
the `sub_task` to specify which specific datasets to fine-tine on. Below is the full list:

| \--task   | \--sub\_task                       | Description                                                                                                                      |
| --------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| summarize | ruby/javascript/go/python/java/php | code summarization task on [CodeSearchNet](https://arxiv.org/abs/1909.09436) data with six PLs                                   |
| concode   | none                               | text-to-code generation on [Concode](https://aclanthology.org/D18-1192.pdf) data                                                 |
| translate | java-cs/cs-java                    | code-to-code translation between [Java and C#](https://arxiv.org/pdf/2102.04664.pdf)                                             |
| refine    | small/medium                       | code refinement on [code repair data](https://arxiv.org/pdf/1812.08693.pdf) with small/medium functions                          |
| defect    | none                               | code defect detection in [C/C++ data](https://proceedings.neurips.cc/paper/2019/file/49265d2447bc3bbfe9e76306ce40a31f-Paper.pdf) |
| clone     | none                               | code clone detection in [Java data](https://arxiv.org/pdf/2002.08653.pdf)                                                        |

For example, if you want to run CodeT5-base model on the code summarization task for Python, you can simply run:

```
python run_exp.py --model_tag codet5_base --task summarize --sub_task python
```

For multi-task training, you can type:

```
python run_exp.py --model_tag codet5_base --task multi_task --sub_task none
```

Besides, you can specify:

```
model_dir: where to save fine-tuning checkpoints
res_dir: where to save the performance results 
summary_dir: where to save the training curves
data_num: how many data instances to use, the default -1 is for using the full data
gpu: the index of the GPU to use in the cluster
``` 

You can also revise the suggested
arguments [here](https://github.com/salesforce/CodeT5/blob/0bf3c0c43e92fcf54d9df68c793ac22f2b60aad4/sh/run_exp.py#L14) or directly customize the [exp_with_args.sh](https://github.com/salesforce/CodeT5/blob/main/sh/exp_with_args.sh) bash file.
Please refer to the argument flags in [configs.py](https://github.com/salesforce/CodeT5/blob/main/configs.py) for the full
available options. The saved training curves in `summary_dir` can be visualized using [tensorboard](https://pypi.org/project/tensorboard/).
Note that we employ one A100 GPU for all fine-tuning experiments.

### How to fine-tune on your own task and dataset?
If you want to fine-tune on your dataset, you can add your own task and sub_task in `configs.py` ([here](https://github.com/salesforce/CodeT5/blob/d27512d23ba6130e089e571d8c3e399760db1c31/configs.py#L11)) and add your data path and the function to read in `utils.py` ([here](https://github.com/salesforce/CodeT5/blob/5bb41e21b07fee73f310476a91ded00e385290d7/utils.py#L103) and [here](https://github.com/salesforce/CodeT5/blob/5bb41e21b07fee73f310476a91ded00e385290d7/utils.py#L149)). The read function can be implemented in `_utils.py` similar to [this one](https://github.com/salesforce/CodeT5/blob/aaf9c4a920c4986abfd54a74f5456b056b6409e0/_utils.py#L213). If your task to add is a generation task, you can simply reuse or customize the `run_gen.py`. For understanding tasks, please refer to `run_defect.py` and `run_clone.py`.

## Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. We welcome PRs!

## Fine-tune on pytorch dataset
```
CUDA_VISIBLE_DEVICES=0 python /remote-home/cchang/project/CodeT5/run_gen.py --do_train --do_eval --do_eval_bleu --do_test --task summarize --sub_task pytorch --model_type codet5 --data_num -1 --num_train_epochs 30 --warmup_steps 1000 --learning_rate 3e-5 --patience 2 --tokenizer_name=Salesforce/codet5-base --model_name_or_path=/remote-home/cchang/project/CodeT5/pretrained_models/codet5_base/ --data_dir /remote-home/cchang/project/CodeT5/data --cache_path saved_models/summarize/pytorch/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/cache_data --output_dir saved_models/summarize/pytorch/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15 --summary_dir tensorboard --save_last_checkpoints --always_save_model --res_dir saved_models/summarize/pytorch/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/prediction --res_fn results/summarize_codet5_base.txt --train_batch_size 4 --gradient_accumulation_steps 12 --eval_batch_size 4 --max_source_length 512 --max_target_length 256 2>&1 | tee saved_models/summarize/pytorch/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/train.log
```