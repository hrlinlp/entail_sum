from collections import defaultdict
import numpy as np

def map_ent_label(labels):
    ent_label_dict = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    new_labels = [[ent_label_dict[label[0]]] for label in labels]
    return new_labels

def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_slice(data, batch_size, sort=True, ent=False):
    if len(data[0]) == 3:
        ent = True
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        if ent:
            labels = [data[i * batch_size + b][2] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]
            if ent:
                labels = [labels[src_id] for src_id in src_ids]

        if ent:
            yield src_sents, tgt_sents, labels
        else:
            yield src_sents, tgt_sents


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
