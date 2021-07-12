# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-12
# @Contact : liaozhi_edo@163.com


"""
    DeepWalk模型
"""

# packages
from walker import RandomWalker
from gensim.models import Word2Vec


class DeepWalk:
    """
    DeepWalk算法
    """
    def __init__(self, graph, num_walks, walk_length, weight):
        """
        初始化

        :param graph: Graph(networkx) 图
        :param num_walks: int 路径数
        :param walk_length: int 路径长度
        :param weight: bool 是否采用加权游走
        """
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walker = RandomWalker(graph, p=1, q=1)
        self.walks = self.walker.random_walk_in_multi_process(num_walks, walk_length, weight=weight)
        self.w2v_model = None
        self.embedding_dict = dict()

    def train(self, embedding_size=64, window_size=5, workers=8, num_iters=10, **kwargs):
        """

        :param embedding_size: int embedding维度
        :param window_size: int 窗口大小
        :param workers: int cpu个数
        :param num_iters: int 迭代
        :param kwargs:
        :return:
        """
        kwargs['sentences'] = self.walks
        kwargs['size'] = embedding_size
        kwargs['window'] = window_size
        kwargs['min_count'] = kwargs.get('min_count', 1)
        kwargs['workers'] = workers
        kwargs['sg'] = 1  # skip gram
        kwargs['hs'] = 1  # DeepWalk use Hierarchical SoftMax
        kwargs['iter'] = num_iters

        model = Word2Vec(**kwargs)
        self.w2v_model = model
        print('Learning embedding vectors done!')

        return model

    def get_embeddings(self):
        """
        获取节点向量

        :return:
            embedding_dict: dict 节点向量
        """
        # 1,判断模型是否训练
        if self.w2v_model is None:
            print("model not train")
            return dict()

        # 2,获取向量
        for node in self.graph.nodes():
            self.embedding_dict[node] = self.w2v_model.wv[node]

        return self.embedding_dict
