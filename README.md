# CCL2022-MCLTC-kk
该仓库为CCL2022-CLTC(多维度汉语学习者文本纠错比赛)赛道三的第一名解决方案，队伍名称为kk。


比赛详情见https://github.com/blcuicall/CCL2022-CLTC

## 1. 环境配置

- 训练推理阶段：requirements_train.txt
- 后处理及模型融合阶段：requirements_postedit.txt

## 2. 预训练模型下载

将预训练模型下载到服务器上，并更新各种可执行文件夹中的BART_DIR变量：

- bart-large-chinese: https://huggingface.co/fnlp/bart-large-chinese

## 3. 数据预处理

训练和验证数据需分别处理成 `.src`  和 `.tgt` 两个文件并放在 `data/raw` 目录下，目录结构如下。

注意：由于参赛过程中使用了两阶段训练，所以有多份训练集和验证集，详情可见技术评测报告。
- 第一阶段训练集是官方训练集，第一阶段验证集是官方验证集。
- 第二阶段以9:1的比例分割官方验证集，分别作为第二阶段训练集和验证集。
- 两个训练阶段都以平均F0.5作为模型挑选标准。

```
data
  └── raw
    ├── train_lang8.src #第一阶段训练集和验证集
    ├── train_lang8.tgt
    ├── valid_lang8.src
    ├── valid_lang8.tgt
    ├── train_vminimal.src #第二阶段训练集和验证集1（使用minimal维度语料）
    ├── train_vminimal.tgt
    ├── valid_vminimal.src
    ├── valid_vminimal.tgt
    ├── train_vfluency.src #第二阶段训练集和验证集2（使用fluency维度语料）
    ├── train_vfluency.tgt
    ├── valid_vfluency.src
    ├── valid_vfluency.tgt
    ├── train_vfluandmin.src #第二阶段训练集和验证集3（混合minimal维度和fluency维度语料）
    ├── train_vfluandmin.tgt
    ├── valid_vfluandmin.src
    └── valid_vfluandmin.tgt
  ├── bpe
  ├── lang8
  ├── vminimal
  ├── vfluency
  └── vfluandmin
```
首先参照`bpe_kk.sh`的示例进行分词，将`data/raw/`的文件中全部分词并生成对应文件。
其次参考`process_kk.sh`的示例进行二值化处理，将`data/bpe/`全部处理并放置到对应文件夹中。
每个文件夹的目录结构如下,以`data/lang8/`为例：
```
data
  ├── raw
  ├── bpe
  └── lang8
    ├── dict.src.txt
    ├── dict.tgt.txt
    ├── train.src-tgt.src.bin
    ├── train.src-tgt.src.idx
    ├── train.src-tgt.tgt.bin
    ├── train.src-tgt.tgt.idx
    ├── valid.src-tgt.src.bin
    ├── valid.src-tgt.src.idx  
    ├── valid.src-tgt.tgt.bin
    ├── valid.src-tgt.tgt.idx  
    └── preprocess.log
```
## 4. 模型训练
### 第一阶段训练
- 请注意`train.py`中 `line639、line748、line754、line755`的代码注释。
- 第一阶段训练使用`train.sh`脚本。
- 本阶段使用了dynamic mask，所以第一阶段训练所能得到的最优模型具有较大随机性。
### 第二阶段训练
- 该阶段必须注释`train.py`中`line388-line412`，关闭动态加噪。
- 请注意`train.py`中 `line639、line748、line754、line755`的代码注释。
- 第二阶段训练同样使用`train.sh`脚本，注意更改DATA_DIR、MODEL_NAME，并添加finetune参数。
## 5. 模型推理
- 请注意`interactive.py`中 `line208、line209、line211、line302、line303、line304`的代码注释。
- 推理使用`interactive.sh`脚本，请修改path参数（sh文件中的DATA_SET是个无效参数，可以无视）。
- 另外，seq2seq模型天然存在unk问题，即使经过文件拼接处理，interactive完成后也只能得到src(无unk)-tgt(有unk)的para文件。
## 6. 模型融合与后处理
- 为彻底去除unk问题，需要使用ChERRANT[^1]计算编辑距离，并将所有包括unk的编辑操作忽略，使用详情可参考文献。
- 在本次比赛中，F0.5作为最终评价指标放大了Precision的比重，所以在做模型融合时，使用了三个非常相似的模型（挑选出第一阶段结束后性能最好的模型，第二阶段分别用minimal验证集、fluency验证集、minimal+fluency验证集训练得到三个模型），将三个模型的编辑操作提取出来，只有三个模型同时出现该编辑操作时，编辑操作才会被最终采纳。



[^1]: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, and Min Zhang. 2022. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction. In Proceedings of NAACL-HLT. ([pdf](https://arxiv.org/pdf/2204.10994.pdf))