import argparse
import glob
import logging
import os
import random


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)

from utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

def load_and_cache_examples(task, tokenizer, ttype='train'):

    train_file = 'train.txt'
    dev_file = 'valid.txt'
    test_file = 'valid.txt'
    data_dir = 'data/codesearch/train_valid/java'
    model_name_or_path = 'microsoft/codebert-base'
    max_seq_length = 128
    model_type = 'roberta'
    local_rank = -1


    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, test_file)
    except:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir, dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, test_file)

        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if model_type in ['xlnet'] else 0)
        if local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset


def evaluate(model, tokenizer, checkpoint=None, prefix="", mode='dev'):

    task_name = 'codesearch'
    output_dir = '.'
    local_rank = -1
    n_gpu = 1
    per_gpu_eval_batch_size = 8
    model_type = 'roberta'
    test_result_dir = 'test_results.tsv'
    device = torch.device("cpu", 0)

    output_mode = output_modes['codesearch']


    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (task_name,)
    eval_outputs_dirs = (output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if (mode == 'dev'):
            eval_dataset = load_and_cache_examples(eval_task, tokenizer, ttype='dev')
        elif (mode == 'test'):
            eval_dataset, instances = load_and_cache_examples(eval_task, tokenizer, ttype='test')

        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = \
                SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:

                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds_label, out_label_ids)
        results.update(result)
        if (mode == 'dev'):
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write('evaluate %s\n' % checkpoint)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        elif (mode == 'test'):
            with open(output_test_file, "w") as writer:
                logger.info("***** Output test results *****")
                all_logits = preds.tolist()
                for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                    instance_rep = '<CODESPLIT>'.join(
                        [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])

                    writer.write(
                        instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))

    return results
