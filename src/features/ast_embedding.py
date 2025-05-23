'''
AST embedding module
This module contains functions to traverse the AST (Abstract Syntax Tree)
and extract features from it.
'''
import json
import numpy as np
from typing import Union, List, Optional, TypedDict
from sklearn.preprocessing import LabelEncoder


# Define a TypedDict for the AST node
class ASTNode(TypedDict):
    '''ASTNode class
        This class represents a node in the Abstract Syntax Tree (AST).
        to store the type, text, name, and children of the node.
    '''
    type: str
    text: Optional[str]
    name: Optional[str]
    children: Union[List['ASTNode'], 'ASTNode']

# Process AST nodes, to convert into a matrix embedding
def traverse_ast(
    node: ASTNode,
    depth=0,
    features=None
) -> list[dict[str,int|str|bool]]:
    '''
    Recursively traverse the AST and extract features.
    Args:
        node (dict): The AST node to process.
        depth (int): The current depth in the AST.
        features (list): The list to store features.
    Returns:
        list[dict[str,int|str|bool]]: A list of features extracted from the AST.
    '''
    if features is None:
        features = []
    if not isinstance(node, dict):
        return features
    node_type = node.get("type", "UNK")

    children = node.get("children", [])

    # Normalize children to a list
    if isinstance(children, dict):
        children = [children]
    elif not isinstance(children, list):
        children = []

    # Save the features for the current node
    features.append({
        "type": node_type,
        "depth": depth,
        "children_count": len(children),
        "is_leaf": len(children) == 0
    })

    for child in children:
        traverse_ast(child, depth + 1, features)
    return features


def read_ast_from_file(
    file_path: str
) -> dict:
    '''
    Read the AST from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
        Returns:
        dict: The AST data.
    '''
    with open(file_path, 'r') as f:

        ast_data = json.load(f)
    return ast_data

def encode_features(
    features: list[dict[str, int|str|bool]]
) -> list[dict[str, int]]:
    '''
    Encode the features into a matrix.
    Args:
        features (list): The list of features to encode.
    Returns:
        np.array: The encoded features as a matrix.
    '''
    type_encoder = LabelEncoder()
    all_types = [feature["type"] for feature in features]
    type_encoder.fit(all_types)
    type_ids = type_encoder.transform(all_types)

    # Each node will have 4 features:
    # 1. Type ID
    # 2. Depth
    # 3. Number of children
    # 4. Is leaf (1 if leaf, 0 otherwise)
    matrix = np.zeros((len(features), 4), dtype=int)
    for i, feature in enumerate(features):
        matrix[i, 0] = type_ids[i]
        matrix[i, 1] = feature["depth"]
        matrix[i, 2] = feature["children_count"]
        matrix[i, 3] = int(feature["is_leaf"])

    return matrix
