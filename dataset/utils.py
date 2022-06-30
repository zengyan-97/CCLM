import re
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm

from utils.hdfs_io import hexists, hcopy, hopen


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption_raw = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption


def write_jsonl(result: list, wpath: str):
    if wpath.startswith('hdfs'):
        with hopen(wpath, 'w') as f:
            for res in result:
                to_write = json.dumps(res, ensure_ascii=False) + '\n'
                f.write(to_write.encode())
    else:
        with open(wpath, 'wt') as f:
            for res in result:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')


def read_jsonl(rpath: str):
    result = []
    if rpath.startswith('hdfs'):
        with hopen(rpath, 'r') as f:
            for line in f:
                result.append(json.loads(line.decode().strip()))
    else:
        with open(rpath, 'rt') as f:
            for line in f:
                result.append(json.loads(line.strip()))

    return result


def collect_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False, save_result=False, remove_duplicate='', do_not_collect=False):
    assert isinstance(result, list)
    write_jsonl(result, os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                    '%s_rank%d.json' % (filename, utils.get_rank())))
    dist.barrier()

    if do_not_collect:
        return None

    result = []
    final_result_file = ''
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            result += read_jsonl(os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                             '%s_rank%d.json' % (filename, rank)))

        if remove_duplicate:  # for evaluating captioning tasks
            result_new = []
            id_list = set()
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.json' % filename)
            json.dump(result, open(final_result_file, 'w'), ensure_ascii=False, indent=4)
            print('result file saved to %s' % final_result_file)
            if write_to_hdfs:
                hcopy(final_result_file, os.path.join(hdfs_wdir, '%s.json' % filename))
                print('result file saved to %s' % os.path.join(hdfs_wdir, '%s.json' % filename))

    dist.barrier()

    return final_result_file if save_result else result


def collect_tensor_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False):
    wpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, utils.get_rank()))
    torch.save(result, wpath)
    if write_to_hdfs:
        hcopy(wpath, hdfs_wdir)

    dist.barrier()

    result = []
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            rpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, rank))
            if write_to_hdfs:
                hcopy(os.path.join(hdfs_wdir, '%s_rank%d.pth' % (filename, rank)), rpath)

            result += torch.load(rpath)

    dist.barrier()

    return result
