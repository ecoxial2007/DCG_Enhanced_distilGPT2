import csv
import json
import ast
import os.path
import torch
import h5py

with open('../../dataset/iu_x-ray/node_mapping.json', 'r') as f:
    node_mapping = json.load(f)


def parse_edges(edge_strings, node_mapping):
    """
    Parses edge strings to determine which nodes are connected.

    :param edge_strings: List of edge strings (e.g., 'no effusion-pleural')
    :param node_mapping: Dictionary mapping node names to their indices.
    :return: A list of tuples representing edges.
    """
    edges = []
    for edge_string in edge_strings:
        nodes = edge_string.split('-')
        if len(nodes) == 2:
            node1, node2 = nodes
            if node1 in node_mapping and node2 in node_mapping:
                edges.append((node_mapping[node1], node_mapping[node2]))
    return edges


def create_graph(edge_strings, num_nodes):
    """
    Creates a graph given node features and edge descriptions.

    :param edge_strings: List of strings representing edges.
    :param num_nodes: Total number of nodes in the graph.
    :return: Adjacency matrix of the graph.
    """
    # Assume a mapping from node names to indices

    # Parse edges
    edges = parse_edges(edge_strings, node_mapping)

    # Initialize adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Populate adjacency matrix based on edges
    for node1, node2 in edges:
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1

    return adjacency_matrix


def parse_and_extract_unique_strings(string):
    """
    Parses a string representation of a list and extracts unique strings.

    :param string: A string representation of a list.
    :return: A set of unique strings.
    """

    def extract_unique_strings(nested_list):
        # 扁平化嵌套列表并去重
        return list(set(item for sublist in nested_list for item in sublist))

    parsed_list = ast.literal_eval(string)
    return extract_unique_strings(parsed_list)


# 读取CSV文件并创建一个以study_id为键的字典
csv_info = {}
with open('new_iuxray.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        study_id = str(row['study_id'])
        csv_info[study_id] = row

# 读取原始的JSON列表
with open('./dataset/iu_x-ray/annotation_top5.json', 'r') as jsonfile:
    json_list = json.load(jsonfile)

organs_list = []
diseases_list = []

top_k = 1

# 更新JSON列表中的信息
for item in json_list['train'] + json_list['test'] + json_list['val']:
    concat_disease = []
    count = 0

    for top_k_image_path in item['top_k_image_path']:
        top_k_study_id = top_k_image_path.split('/')[-2]
        if top_k_study_id in csv_info:
            # item['organs_list'] = parse_and_extract_unique_strings(csv_info[study_id]['organs_list'])
            # item['diseases_list'] = parse_and_extract_unique_strings(csv_info[study_id]['diseases_list'])
            item['disease_type'] = parse_and_extract_unique_strings(csv_info[top_k_study_id]['disease_type'])
            item['normal_type'] = parse_and_extract_unique_strings(csv_info[top_k_study_id]['normal_type'])
            concat_disease += item['disease_type'] + item['normal_type']
            count += 1

        if count >= top_k: break

    concat_disease = list(set(concat_disease))
    adjacency_matrix = create_graph(concat_disease, 191)

    file_name = f"{item['id']}.h5"
    file_path = os.path.join('dataset/iu_x-ray/adjacency_matrix_191', file_name)
    print(file_path, adjacency_matrix.shape)
    with h5py.File(file_path, 'w') as h5file:
        dset = h5file.create_dataset('adj_matrix', data=adjacency_matrix.numpy())
