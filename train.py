#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import random

from tqdm import tqdm
from multiprocessing import Pool
from opencc import OpenCC
from collections import Counter

import argparse
import logging
import math
import os
import sys
import time
import fileinput
from collections import namedtuple
from fairseq.token_generation_constraints import pack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from typing import Any, Callable, Dict, List, Optional, Tuple
sys.path.append(".../metrics")
import wandb
from modules.tokenizer import Tokenizer
from modules.annotator import Annotator

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from utils.wandb_state import State

state = State()
list_chtarget = []
list_chfreq = []
list_pyfreq = []
list_tfidf = []
list_all = [10]*21128


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    experiment_name = "{}-{}-{}".format(
        cfg.common.seed,
        cfg.dataset.max_tokens,
        str(int(time.time())),
    )
    logger.info("Set wandb")
    wandb.init(
        project="hr-gec",
        name=experiment_name,
        entity='jinhao-jiang',
    )

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    if 'reverse' in task.cfg.data:
        logger.info("______________reverse______________")
    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    with open('/.../id2chfreq.txt', 'r', encoding='utf8') as r1:
        with open('/.../id2pyfreq.txt', 'r', encoding='utf8') as r2:
            with open('/.../id2tfidf.txt', 'r', encoding='utf8') as r3:
                for line1, line2, line3 in zip(r1, r2, r3):
                    t1 = line1.strip().split(' ')[1]
                    t2 = line2.strip().split(' ')[1]
                    t3 = line3.strip().split(' ')[1]
                    if int(t1) > 0:
                        list_chtarget.append(0)
                    else:
                        list_chtarget.append(100)
                    list_chfreq.append(int(t1))
                    list_pyfreq.append(int(t2))
                    list_tfidf.append(float(t3))

    for i in range(0,107):
        list_all[i] = 0

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the，在load_checkpoint里判断第几个epoch和加载train_dataset
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        # for sample in samples:
        #     for i in range(len(sample['net_input']['src_tokens'])):
        #         for j in range(len(sample['net_input']['src_tokens'][0])):
        #             print(str(sample['net_input']['src_tokens'][i][j]), end=" ")
        #         print('\n\n\n')

        # for sample in samples:
        #     #挑出要被改错的句子序号
        #     all_index = [i for i in range(len(sample['net_input']['src_tokens']))]
        #     change_select = random.sample(all_index, int(len(all_index)*0.1))
        #     change_select.sort()
        #
        #     scheme_select = [random.randint(1, 1) for i in range(len(change_select))]
        #     # pad_mask_or_at = [random.randint(0, 1) for i in range(len(change_select))]
        #     pad_mask = torch.where(sample['net_input']['src_tokens'] == 0, 0, 1)
        #
        #     #某一概率下部分加噪
        #     for sidx, s in zip(change_select, scheme_select):
        #         index_102 = (sample['net_input']['src_tokens'][sidx] == 102).nonzero(as_tuple=True)[0].tolist()
        #         sentence_len = sample['net_input']['src_lengths'][sidx]
        #         if sentence_len > 4:
        #             token_index = [i for i in range(1, sentence_len)]
        #             token_change = random.sample(token_index, 4)
        #
        #             for token in token_change:
        #                 if token not in index_102:
        #                     if s == 0:
        #                         replace_token = 1
        #                     elif s == 1:
        #                         replace_token = list(WeightedRandomSampler(list_chtarget, 1, replacement=True))[0]
        #                     elif s == 2:
        #                         replace_token = list(WeightedRandomSampler(list_tfidf, 1, replacement=True))[0]
        #                     elif s == 3:
        #                         replace_token = list(WeightedRandomSampler(list_pyfreq, 1, replacement=True))[0]
        #                     elif s == 4:
        #                         replace_token = list(WeightedRandomSampler(list_chfreq, 1, replacement=True))[0]
        #                     elif s == 5:
        #                         replace_token = list(WeightedRandomSampler(list_all, 1, replacement=True))[0]
        #                     else:
        #                         continue
        #                     sample['net_input']['src_tokens'][sidx][token] = replace_token

            # #某一概率下全部加噪
            # select_tensor = torch.randint(0, 99, size=sample['net_input']['src_tokens'].size())
            # select_tensor[:, 0] = 31
            # #类型指定，randint(0,0)就是pad，(1,1)就是random
            # scheme_select = [random.randint(0,4) for i in range(len(sample['net_input']['src_tokens']))]
            # pad_mask = torch.where(sample['net_input']['src_tokens'] == 0, 0, 1)
            # # 改一下总体概率
            # select_mask_1 = torch.where(select_tensor < 10, 1, 0) * pad_mask  # 需要被替换的地方为1
            # for sidx, s in enumerate(scheme_select):
            #     index_102 = (sample['net_input']['src_tokens'][sidx] == 102).nonzero(as_tuple=True)[0].tolist()
            #     index_1 = (select_mask_1[sidx] == 1).nonzero(as_tuple=True)[0].tolist()
            #     for tidx in index_1:
            #         if tidx not in index_102:
            #             if s == 0:
            #                 replace_token = 0
            #             elif s == 1:
            #                 replace_token = list(WeightedRandomSampler(list_chtarget, 1, replacement=True))[0]
            #             elif s == 2:
            #                 replace_token = list(WeightedRandomSampler(list_chfreq, 1, replacement=True))[0]
            #             elif s == 3:
            #                 replace_token = list(WeightedRandomSampler(list_pyfreq, 1, replacement=True))[0]
            #             else:
            #                 continue
            #             sample['net_input']['src_tokens'][sidx][tidx] = replace_token

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            state.record_each_step(loss=log_output['loss'], nll_loss=log_output['nll_loss'])
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = inferdev_and_save(cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch)
        # valid_losses, should_stop = validate_and_save(
        #     cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        # )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop

def inferdev_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    f_flu = 0
    valid_losses = [None]
    #begin infer
    if do_validate:
        cfg.generation.beam = 5
        cfg.dataset.batch_size = 64
        cfg.interactive.buffer_size = 64
        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[trainer.model.max_positions()]
        )
        total_translate_time = 0
        start_time = time.time()
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        trainer.model.prepare_for_inference_(cfg)
        #这边应该能优化
        generator = task.build_generator([trainer.model], cfg.generation)
        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)

        def encode_fn(x):
            if tokenizer is not None:
                x = tokenizer.encode(x)
            if bpe is not None:
                x = bpe.encode(x)
            return x

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x
        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(cfg.generation.replace_unk)
        if cfg.interactive.buffer_size > 1:
            logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
        logger.info("NOTE: hypothesis and token scores are output in base 2")

        save_file = cfg.checkpoint.save_dir + '/devresult-{}.para'.format(epoch_itr.epoch)
        start_id = 0
        with open(save_file, 'a',
                  encoding='utf-8') as w:
            cfg.interactive.input = '.../valid_fluency_test.src'
            for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
                results = []
                for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
                    bsz = batch.src_tokens.size(0)
                    src_tokens = batch.src_tokens.cuda()
                    src_lengths = batch.src_lengths.cuda()
                    constraints = batch.constraints
                    sample = {
                        "net_input": {
                            "src_tokens": src_tokens,
                            "src_lengths": src_lengths,
                        },
                    }
                    translate_start_time = time.time()
                    translations = task.inference_step(
                        generator, trainer.model, sample, constraints=constraints
                    )
                    translate_time = time.time() - translate_start_time
                    total_translate_time += translate_time
                    list_constraints = [[] for _ in range(bsz)]
                    for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                        constraints = list_constraints[i]
                        results.append(
                            (
                                start_id + id,
                                src_tokens_i,
                                hypos,
                                {
                                    "constraints": constraints,
                                    "time": translate_time / len(translations),
                                },
                            )
                        )

                # sort output to match input order
                for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                    src_str = ""
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                        # print("S-{}\t{}".format(id_, src_str))
                        w.write(str(id_))
                        res_src = src_str.replace(' ','')
                        w.write('\t'+res_src)
                        # print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                        for constraint in info["constraints"]:
                            print(
                                "C-{}\t{}".format(
                                    id_,
                                    tgt_dict.string(constraint, cfg.common_eval.post_process),
                                )
                            )

                    # Process top predictions
                    for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo["tokens"].int().cpu(),
                            src_str=src_str,
                            alignment=hypo["alignment"],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=cfg.common_eval.post_process,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                        detok_hypo_str = decode_fn(hypo_str)
                        score = hypo["score"] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        # print("H-{}\t{}\t{}".format(id_, score, hypo_str))
                        # detokenized hypothesis
                        # print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
                        res_hypo = detok_hypo_str.replace(' ','')
                        if 'reverse' in task.cfg.data:
                            res_hypo = ''.join(reversed(res_hypo))
                            res_hypo.replace(']KNU[','[UNK]')
                            w.write('\t'+res_hypo+'\n')
                        else:
                            w.write('\t'+res_hypo+'\n')
                        # print(
                        #     "P-{}\t{}".format(
                        #         id_,
                        #         " ".join(
                        #             map(
                        #                 lambda x: "{:.4f}".format(x),
                        #                 # convert from base e to base 2
                        #                 hypo["positional_scores"].div_(math.log(2)).tolist(),
                        #             )
                        #         ),
                        #     )
                        # )
                        if cfg.generation.print_alignment:
                            alignment_str = " ".join(
                                ["{}-{}".format(src, tgt) for src, tgt in alignment]
                            )
                            # print("A-{}\t{}".format(id_, alignment_str))

                # update running id_ counter
                start_id += len(inputs)

        logger.info(
            "Total time: {:.3f} seconds; translation time: {:.3f}".format(
                time.time() - start_time, total_translate_time
            )
        )

        #calculate f0.5
        save_file_path = save_file.replace('.para', '_final.para')
        with open(save_file_path, 'w', encoding='utf8') as w:
            with open(save_file, 'r', encoding='utf8') as res:
                with open('/.../yaclc-fluence_dev_astest.src', 'r', encoding='utf8') as test:
                    for src, res in zip(test, res):
                        src = src.strip()
                        res = res.strip()
                        id_, _, tgt_ = res.split('\t')
                        w.write(str(id_) + '\t' + src + '\t' + tgt_ + '\n')
        f_flu = evaluate(save_file_path, "/.../yaclc-fluency_dev.m2", cfg)
        f_min = evaluate(save_file_path, "/...yaclc-minimal_dev.m2", cfg)
        valid_losses, stats = validate(cfg, trainer, task, epoch_itr, valid_subsets)
        state.record_each_step(f_flu=f_flu, f_min=f_min, valid_loss=valid_losses, valid_ppl=stats['ppl'])

    should_stop |= should_stop_early(cfg, f_flu)

    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, f_flu
        )
    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses, stats


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )

def evaluate(res_file, ref_file, cfg):
    tokenizer = Tokenizer('char', cfg.model.device_id, False)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default('char',None)
    lines = open(res_file,"r").read().strip().split("\n")

    hyp_m2 = []
    count = 0
    sentence_set = set()
    sentence_to_tokenized = {'': ''}
    for line in lines:
        sent_list = line.split("\t")[1:]
        # sent_list = line.split("\t")
        for idx, sent in enumerate(sent_list):
            sent = "".join(sent.split()).strip()
            if idx >= 1:
                sentence_set.add(cc.convert(sent))
            else:
                sentence_set.add(sent)
    batch = []
    for sent in tqdm(sentence_set):
        count += 1
        if sent:
            batch.append(sent)
        if count % 128 == 0: #128 is args.batch_size in original code
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r
    with Pool(16) as pool:
        for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
            if ret:
                hyp_m2.append(ret)

    ref_m2 = open(ref_file).read().strip().split("\n\n")
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
    best_cats = {}
    # Process each sentence
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        # Simplify the edits into lists of lists
        # if "A1" in sent[0] or "A1" in sent[1] or sent_id in sent_id_cons:
        #     sent_id_cons.append(sent_id)
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], None)
        ref_edits = simplify_edits(sent[1], None)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits)
        ref_dict = process_edits(ref_edits)

        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(src, hyp_dict, ref_dict,
                                              best_dict, sent_id)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
    # Print results
    return print_results(best_dict, best_cats)


def print_results(best, best_cats):

    title = " Span-Based Correction "

    # Print the overall results.
    tp, fp, fn = computeFScore(best["tp"], best["fp"], best["fn"], 0.5)
    print("")
    print('{:=^46}'.format(title))
    print("\t".join(["TP", "FP", "FN", "Prec", "Rec", "F" + str(0.5)]))
    print("\t".join(
        map(str, [best["tp"], best["fp"], best["fn"]] + list(
            computeFScore(best["tp"], best["fp"], best["fn"], 0.5)))))
    print('{:=^46}'.format(""))
    print("")

    return fn

def merge_dict(dict1, dict2):
    for cat, stats in dict2.items():
        if cat in dict1.keys():
            dict1[cat] = [x + y for x, y in zip(dict1[cat], stats)]
        else:
            dict1[cat] = stats
    return dict1

def evaluate_edits(src, hyp_dict, ref_dict, best, sent_id):
    # Store the best sentence level scores and hyp+ref combination IDs
    # best_f is initialised as -1 cause 0 is a valid result.
    best_tp, best_fp, best_fn, best_f, best_hyp, best_ref = 0, 0, 0, -1, 0, 0
    best_cat = {}
    # skip not annotatable sentence
    if len(ref_dict.keys()) == 1:
        ref_id = list(ref_dict.keys())[0]
        if len(ref_dict[ref_id].keys()) == 1:
            cat = list(ref_dict[ref_id].values())[0][0]
            if cat == "NA":
                best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
                return best_dict, best_cat

    # Compare each hyp and ref combination
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            # Get the local counts for the current combination.
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id],
                                                ref_dict[ref_id])
            # Compute the local sentence scores (for verbose output only)
            loc_p, loc_r, loc_f = computeFScore(tp, fp, fn, 0.5)
            # Compute the global sentence scores
            p, r, f = computeFScore(tp + best["tp"], fp + best["fp"],
                                    fn + best["fn"], 0.5)
            # Save the scores if they are better in terms of:
            # 1. Higher F-score
            # 2. Same F-score, higher TP
            # 3. Same F-score and TP, lower FP
            # 4. Same F-score, TP and FP, lower FN
            if (f > best_f) or \
                    (f == best_f and tp > best_tp) or \
                    (f == best_f and tp == best_tp and fp < best_fp) or \
                    (f == best_f and tp == best_tp and fp == best_fp
                        and fn < best_fn):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f, best_hyp, best_ref = f, hyp_id, ref_id
                best_cat = cat_dict
    # Save the best TP, FP and FNs as a dict,
    # and return this and the best_cat dict
    best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
    return best_dict, best_cat

def computeFScore(tp, fp, fn, beta):
    p = float(tp) / (tp + fp) if fp else 1.0
    r = float(tp) / (tp + fn) if fn else 1.0
    f = float(
        (1 + (beta**2)) * p * r) / (((beta**2) * p) + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def compareEdits(hyp_edits, ref_edits):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    cat_dict = {}  # {cat: [tp, fp, fn], ...}

    for h_edit, h_cats in hyp_edits.items():
        # noop hyp edits cannot be TP or FP
        if h_cats[0] == "noop":
            continue
        # TRUE POSITIVES
        if h_edit in ref_edits.keys():
            # On occasion, multiple tokens at same span.
            for h_cat in ref_edits[h_edit]:  # Use ref dict for TP
                tp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][0] += 1
                else:
                    cat_dict[h_cat] = [1, 0, 0]
        # FALSE POSITIVES
        else:
            # On occasion, multiple tokens at same span.
            for h_cat in h_cats:
                fp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][1] += 1
                else:
                    cat_dict[h_cat] = [0, 1, 0]
    for r_edit, r_cats in ref_edits.items():
        # noop ref edits cannot be FN
        if r_cats[0] == "noop":
            continue
        # FALSE NEGATIVES
        if r_edit not in hyp_edits.keys():
            # On occasion, multiple tokens at same span.
            for r_cat in r_cats:
                fn += 1
                # Each dict value [TP, FP, FN]
                if r_cat in cat_dict.keys():
                    cat_dict[r_cat][2] += 1
                else:
                    cat_dict[r_cat] = [0, 0, 1]
    return tp, fp, fn, cat_dict

def process_edits(edits):
    coder_dict = {}
    # Add an explicit noop edit if there are no edits.
    if not edits:
        edits = [[-1, -1, "noop", "-NONE-", 0]]
    # Loop through the edits
    for edit in edits:
        # Name the edit elements for clarity
        start = edit[0]
        end = edit[1]
        cat = edit[2]
        cor = edit[3]
        coder = edit[4]
        # Add the coder to the coder_dict if necessary
        if coder not in coder_dict:
            coder_dict[coder] = {}

        # Optionally apply filters based on args
        # 1. UNK type edits are only useful for detection, not correction.
        if cat == "UNK":
            continue

        if (start, end, cor) in coder_dict[coder].keys():
            coder_dict[coder][(start, end, cor)].append(cat)
        else:
            coder_dict[coder][(start, end, cor)] = [cat]


    return coder_dict

def simplify_edits(sent, max_answer_num):
    out_edits = []
    # Get the edit lines from an m2 block.
    edits = sent.split("\n")
    # Loop through the edits
    for edit in edits:
        # Preprocessing
        if edit.startswith("A "):
            edit = edit[2:].split("|||")  # Ignore "A " then split.
            span = edit[0].split()
            start = int(span[0])
            end = int(span[1])
            cat = edit[1]
            cor = edit[2].replace(" ", "")
            coder = int(edit[-1])
            out_edit = [start, end, cat, cor, coder]
            out_edits.append(out_edit)
    # return [edit for edit in out_edits if edit[-1] in [0,1]]
    if max_answer_num is None:
        return out_edits
    elif max_answer_num == 1:
        return [edit for edit in out_edits if edit[-1] == 0]
    else:
        return [
            edit for edit in out_edits
            if edit[-1] in list(range(max_answer_num))
        ]

def annotate(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]

    source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            target = "".join(target.strip().split())
            target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[
                source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()

    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )
    # cfg.common.cpu = True
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
