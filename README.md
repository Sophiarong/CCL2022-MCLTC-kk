# CCL2022-MCLTC-kk
该仓库为CCL2022-CLTC(多维度汉语学习者文本纠错比赛)赛道三的第一名解决方案，队伍名称为kk。


比赛详情见https://github.com/blcuicall/CCL2022-CLTC

## 1. 环境配置

- pytorch >= 1.8.0
- fariseq

```shell
git clone https://github.com/pytorch/fairseq
cd fairseq
git reset --hard 06c65c82973969
pip install --editable ./
```

- transformers >= 4.7.0
- apex (optional)

## 2. 预训练模型下载

将预训练模型放到 `pretrained-models/` 目录下，需要的模型为：

- bart-large-chinese: https://huggingface.co/fnlp/bart-large-chinese

## 3. 数据预处理

训练和验证数据需分别处理成 `.src`  和 `.tgt` 两个文件并放在 `data/raw` 目录下，目录结构如：

```
data
  └── raw
    ├── train.src
    ├── train.tgt
    ├── valid.src
    └── valid.tgt
  ├── bpe
  └── processed
```

然后运行 `data_process.sh` 进行数据预处理，预处理后的数据放在 `data/raw/processed` 中。

注意：由于参赛过程中使用了两阶段训练，所以有多份训练集和验证集，详情可见技术评测报告。
- 第一阶段训练集是官方训练集，第一阶段验证集是官方验证集。
- 第二阶段以9:1的比例分割官方验证集，分别作为第二阶段训练集和验证集。
- 两个训练阶段都以平均F0.5作为模型挑选标准。

## 4. 模型训练

- 第一阶段训练使用 `train.sh` 脚本。
- 第二阶段训练同样使用`train.sh` 脚本，但是需要添加finetune参数。
- train.py中存在少量读取文件的常量（主要是在F0.5的计算阶段，模型希望能够在每轮训练结束后得到F0.5,以此为标准挑选checkpoint)，未来得及优化（尴尬挠头），请谨慎避坑。
- train.py中372行的大段注释用于第一阶段的官方训练集动态加噪，需要在第二阶段训练过程中关闭。

## 5. 模型融合与模型后处理
- 首先使用脚本 `interactive.sh`获得推断结果。
- sh文件中的DATA_SET是个无效参数，实际输入在interactive.py中的常量。
- 需要注意，seq2seq模型天然存在unk问题，即使经过文件拼接处理，interactive完成后也只能得到src(无unk)-tgt(有unk)的para文件。
- 为彻底去除unk问题，需要使用ChERRANT[^1]计算编辑距离，并将所有包括unk的编辑操作忽略，使用详情可参考文献。
- 在本次比赛中，F0.5作为最终评价指标放大了Precision的比重，所以在做模型融合时，使用了三个非常相似的模型（挑选出第一阶段结束后性能最好的模型，第二阶段分别用minimal验证集、fluency验证集、minimal+fluency验证集训练得到三个模型），将三个模型的编辑操作提取出来，只有三个模型同时出现该编辑操作时，编辑操作才会被最终采纳。



[^1]: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, and Min Zhang. 2022. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction. In Proceedings of NAACL-HLT. ([pdf](https://arxiv.org/pdf/2204.10994.pdf))