# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:44:05 2022
@author: DingJiCai
"""
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

input_sentences = ['苹果公司今年利润大幅下滑', '国产手机销量远超苹果手机',
                   '去年苹果公司又发布了新款手机', '库克是苹果的CEO',
                   '苹果的股价又跌了', '这个苹果又大又甜',
                   '苹果好吃还不贵', '小孩爱喝苹果汁',
                   '中国苹果出口到非洲', '红富士苹果在中国热销']
index_char = [0, 8, 2, 3, 0, 2, 0, 4, 2, 3]
word_vecs = np.zeros((10, 768), dtype=float)
print(word_vecs.shape)
tokenizer = BertTokenizer.from_pretrained('chinese_wwm_ext_pytorch')
model = BertModel.from_pretrained('chinese_wwm_ext_pytorch')

for i, s in enumerate(list(input_sentences)):
    input_IDs = torch.tensor(tokenizer.encode(s)).unsqueeze(0)  # Batch size 1
    outputs = model(input_IDs)
    sequence_output = outputs[0]
    print(i, sequence_output.shape)
    word_vecs[i, :] = sequence_output.detach().numpy()[:, index_char[i]+1, :] + \
                      sequence_output.detach().numpy()[:, index_char[i]+2, :]
print(word_vecs.shape)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2).fit_transform(word_vecs)
fig = plt.figure(figsize=(10,5))
for i in  range(len(pca)):
    if i<=4:
        plt.scatter(pca[i, 0], pca[i, 1], c='r')
    else:
        plt.scatter(pca[i, 0], pca[i, 1], c='b')
plt.show()