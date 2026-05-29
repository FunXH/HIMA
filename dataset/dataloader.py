
from .base import BaseDataset
import os
import numpy as np
import scipy.io as scio


def split_data(captions, indexs, labels, query_num=5000, train_num=10000, seed=None):
    np.random.seed(seed=seed)
    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]
    
    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]
    
    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    return split_indexs, split_captions, split_labels

def dataloader(captionFile: str,
                indexFile: str,
                labelFile: str,
                maxWords=32,
                imageResolution=224,
                query_num=5000, 
                train_num=10000, 
                seed=None,
                npy=False):
    if captionFile.endswith("mat"):
        captions = scio.loadmat(captionFile)["caption"]
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
    if not npy:
        indexs = scio.loadmat(indexFile)["index"]
    else:
        indexs = np.load(indexFile, allow_pickle=True)
    labels = scio.loadmat(labelFile)["category"]
    # for item in ['__version__', '__globals__', '__header__']:
    #     captions.pop(item)
    #     indexs.pop(item)
    #     labels.pop(item)
    
    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num, train_num=train_num, seed=seed)

    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1], maxWords=maxWords, imageResolution=imageResolution, npy=npy)
    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0], maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2], maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)

    # 从原始检索集数据中采样 Target Data
    rng = np.random.default_rng(seed)  # 使用相同的种子保证可复现性
    target_size = query_data.__len__()  # 与查询集大小一致

    # 获取原始检索集的全部索引（split_captions[2]长度即原始检索样本数）
    retrieval_size = len(split_captions[2])
    if retrieval_size < target_size:
        raise ValueError("检索集样本不足以采样目标集！")
    # 生成随机不重复索引（从原始检索数据中采样）
    sample_indices = rng.choice(retrieval_size, size=target_size, replace=False)

    # 根据索引构造新数据集
    target_captions = split_captions[2][sample_indices]
    target_indexs = split_indexs[2][sample_indices]
    target_labels = split_labels[2][sample_indices]

    target_data = BaseDataset(
        captions=target_captions,
        indexs=target_indexs,
        labels=target_labels,
        maxWords=maxWords,
        imageResolution=imageResolution,
        is_train=False,
        npy=npy
    )

    return train_data, query_data, retrieval_data, target_data
    

