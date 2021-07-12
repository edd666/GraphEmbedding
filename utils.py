# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-12
# @Contact : liaozhi_edo@163.com


"""
    Utils
"""

# packages
import networkx as nx
from collections import Counter


def make_item_time_pair(df):
    """
    构建用户行为序列,形如:

    userid: [(itemid, time)] 按时间小到大从左到右

    :param df: 行为数据
    :return:
    """
    return list(zip(df['feedid'], df['date_']))


def build_weighted_graph(user_item_time_dict):
    """
    基于用户行为序列构建有向加权图

    注意: 在构图过程中,过滤掉了只有一次交互行为的用户,即len(item_time_list)=1.

    :param user_item_time_dict: dict 用户行为序列
    :return:
        graph: DiGraph(networkx) 物品关系图
    """
    # 1,获取边集
    edges = []
    edges_ = []  # 中间变量
    for item_time_list in user_item_time_dict.values():
        tmp_edge = [(str(item_time_list[idx][0]), str(item_time_list[idx + 1][0])) for idx in
                    range(len(item_time_list) - 1)]  # 过滤只有一条行为的用户
        edges_.extend(tmp_edge)
    # 权重
    for edge, weight in Counter(edges_).items():
        edges.append((edge[0], edge[1], weight))

    # 2,加权有向图
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)

    return graph

