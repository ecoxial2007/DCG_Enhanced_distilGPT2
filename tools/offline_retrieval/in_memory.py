import json
import faiss
from tqdm import tqdm
import time
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import multiprocessing


def process_file(feature_path, mmap_vkeys, index):
    image_feature = h5py.File(feature_path, 'r')
    image_global_feature = np.array(image_feature['feature'])
    mmap_vkeys[index] = normalize(image_global_feature, axis=1).squeeze(axis=0)




def process_files_batch(args):
    feature_paths_batch, mmap_vkeys, start_index = args
    for i, feature_path in enumerate(feature_paths_batch):
        process_file(feature_path, mmap_vkeys, start_index + i)



class FeatureMemory(nn.Module):
    def __init__(self, memo_path):
        super().__init__()
        self.memo_path = memo_path
        self.d_output = 512
        self.top_k = 5

    def initial_memory(self):
        start = time.time()
        with open(self.memo_path, 'r') as f:
            datas = json.load(f)

        feature_paths = list(datas.keys())
        self.feature_paths = feature_paths
        self.reports = list(datas.values())

        self.memory_size = len(feature_paths)
        num_processes = multiprocessing.cpu_count()
        chunk_size = len(feature_paths) // num_processes

        feature_paths_batches = [feature_paths[i:i + chunk_size] for i in range(0, len(feature_paths), chunk_size)]

        # 创建内存映射文件
        # 假设的特征维度，需要根据您的数据调整
        feature_dim_vkeys = self.d_output
        if not os.path.exists('./temp/'):
            os.mkdir('./temp/')
        try:
            mmap_vkeys = np.memmap('./temp/mmap_vkeys.dat', dtype='float32', mode='r', shape=(self.memory_size, feature_dim_vkeys))
        except:
            mmap_vkeys = np.memmap('./temp/mmap_vkeys.dat', dtype='float32', mode='w+', shape=(self.memory_size, feature_dim_vkeys))

            for i, batch in tqdm(enumerate(feature_paths_batches), total=len(feature_paths_batches), desc="H5 Loading schedule"):
                process_files_batch((batch, mmap_vkeys, i * chunk_size))


        print('Finished Loading H5 files: ', time.time() - start)
        print(mmap_vkeys.shape)
        # 使用内存映射文件进行后续处理
        self.vkeys_np = mmap_vkeys


        d_v = self.vkeys_np.shape[1]
        quantizer_v = faiss.IndexFlatIP(d_v)

        nlist = 100

        # 使用量化器创建 IndexIVFFlat 索引
        self.vindex = faiss.IndexIVFFlat(quantizer_v, d_v, nlist, faiss.METRIC_INNER_PRODUCT)

        # 训练索引
        if not self.vindex.is_trained:
            self.vindex.train(self.vkeys_np)

        # 添加向量到索引
        self.vindex.add(self.vkeys_np)
        print('Finished Initialize Faiss Key:', time.time() - start)
        print('Multi-process Time:', time.time() - start)




    def max_sum_dist(self, query, keys):
        similarity_scores = torch.einsum('bpd,bktd->bkpt', F.normalize(query, dim=-1), F.normalize(keys, dim=-1))

        max_scores, _ = torch.max(similarity_scores, dim=3)
        final_scores = F.softmax(torch.sum(max_scores, dim=2), dim=1)

        return final_scores.unsqueeze(-1).unsqueeze(-1)


    def retrieve_by_faiss(self, vquery, k):
        vquery_np = vquery.numpy() if isinstance(vquery, torch.Tensor) else vquery
        print(vquery_np.shape)
        v_D, v_I = self.vindex.search(vquery_np, k)
        return v_D, v_I

    def retrieve_and_process(self, vquery):
        """
        根据查询特征检索并处理局部特征。

        :param query_feature: 查询特征。
        :return: 处理后的特征。
        """
        vquery = F.normalize(vquery, p=2, dim=1).cpu()
        batch_size = vquery.shape[0]

        v_similarities, v_top_k_indices = self.retrieve_by_faiss(vquery,  self.top_k)
        return v_similarities, v_top_k_indices

    def load_feature(self, feature_path):
        with h5py.File(feature_path, 'r') as hf:
            feature = np.array(hf['feature'])
        return torch.tensor(feature)  # Assuming the feature needs to be a batch



    def process_annotations(self, annotation_path):
        splits = ['test']
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        del data['train']
        for split in splits:
            for value in data[split]:
                top_k_image_paths = []
                for image_path in value['image_path']:
                    # image_path = os.path.join('./dataset/mimic_cxr/images', image_path)
                    feature_path = image_path.replace('images', 'image_features')\
                        .replace('.jpg', '.h5').replace('.png', '.h5')
                    print(feature_path)
                    feature = self.load_feature(feature_path)
                    v_similarities, v_top_k_indices = self.retrieve_and_process(feature)

                    # Retrieve the paths of the top-k similar features
                    top_k_paths = {self.feature_paths[i]: self.reports[i] for i in v_top_k_indices[0]}
                    top_k_image_paths.append(top_k_paths)

                value['top_k_image_path'] = top_k_image_paths

        with open(f'./dataset/mimic_cxr/annotation_top{self.top_k}.json', 'w') as fff:
            json.dump(data, fff, indent=4)


if __name__ == '__main__':
    memo = FeatureMemory('./dataset/memory_list.json')
    memo.initial_memory()
    memo.process_annotations('./dataset/mimic_cxr/annotation.json')