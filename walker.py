# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-12
# @Contact : liaozhi_edo@163.com


"""
    游走算法
"""

# packages
import random
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


class RandomWalker:
    """
    随机游走算法
    """
    def __init__(self, graph, p=1, q=1):
        """
        初始化

        :param graph: Graph(networkx) 图
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes.
        :return:
        """
        self.graph = graph
        self.p = p
        self.q = q

    def random_walk(self, start_node, walk_length):
        """
        随机游走算法

        :param start_node: int 起始点
        :param walk_length: int 路径长度
        :return:
            walk: list 路径
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break

        return walk

    def random_walk_with_weight(self, start_node, walk_length):
        """
        加权随机游走

        :param start_node: int 起始点
        :param walk_length: int 路径长度
        :return:
            walk: list 路径
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                weights = [self.graph[cur][nbr].get('weight', 1.0) for nbr in cur_nbrs]
                norm = sum(weights)
                probs = [float(weight) / norm for weight in weights]
                walk.append(np.random.choice(cur_nbrs, size=1, p=probs)[0])
            else:
                break

        return walk

    def random_walk_in_batch(self, args):
        """
        随机游走算法--批量

        :param args: list 参数 [(node, walk_length)]
        :return:
        """
        # 1,随机游走
        walks = []
        for arg_tuple in args:
            start_node, walk_length, weight = arg_tuple
            if weight:
                walks.append(self.random_walk_with_weight(start_node, walk_length))  # 加权游走
            else:
                walks.append(self.random_walk(start_node, walk_length))  # 非加权游走

        return walks

    def random_walk_in_multi_process(self, num_walks, walk_length, weight=True):
        """
        随机游走算法--多进程版本

        :param num_walks: int 路径数
        :param walk_length: int 路径长度
        :param weight: bool 是否采用加权游走
        :return:
             walks: list 游走的路径
        """
        # 1,初始化
        nodes = list(self.graph.nodes())
        num_cpus = cpu_count() // 3 if len(nodes) > 500 * cpu_count() else 1

        # 2,参数划分
        args_list_ = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                args_list_.append((node, walk_length, weight))

        # 切分
        args_list = []
        batch_size = num_walks * len(nodes) // num_cpus
        for idx in range(0, len(args_list_), batch_size):
            args_list.append(args_list_[idx:idx + batch_size])

        # 3,多进程随机游走
        total_walks = []
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for walks in executor.map(self.random_walk_in_batch, args_list):
                print('a walker has generated {} sentence'.format(len(walks)))
                total_walks = total_walks + walks

        return total_walks
