# Full traverse_ast.py — Enhanced AST parsing and encoding
import json
from typing import Union, List, Optional, TypedDict
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ASTNode(TypedDict):
    type: str
    token: Optional[str]
    text: Optional[str]
    name: Optional[str]
    children: Union[List['ASTNode'], 'ASTNode']

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "var", "record", "sealed",
    "permits", "non-sealed", "yield"
}

def traverse_ast(node: ASTNode, depth=0, features=None, parent=None, sibling_index=0, num_siblings=0):
    if features is None:
        features = []
    if not isinstance(node, dict):
        return features

    node_type = node.get("type", "UNK")
    token = node.get("token") or node.get("value") or node.get("text") or ""

    children = node.get("children", [])
    if isinstance(children, dict):
        children = [children]
    elif not isinstance(children, list):
        children = []

    features.append({
        "type": node_type,
        "token": token,
        "depth": depth,
        "children_count": len(children),
        "is_leaf": len(children) == 0,
        "token_length": len(token),
        "token_is_keyword": token in JAVA_KEYWORDS,
        "sibling_index": sibling_index,
        "num_siblings": num_siblings,
    })

    for idx, child in enumerate(children):
        traverse_ast(child, depth + 1, features, node, idx, len(children))
    return features

def read_ast_from_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_features(features, max_nodes=500):
    all_types = [f["type"] for f in features]
    all_tokens = [f["token"] or "ε" for f in features]

    type_encoder = LabelEncoder()
    token_encoder = LabelEncoder()
    type_ids = type_encoder.fit_transform(all_types)
    token_ids = token_encoder.fit_transform(all_tokens)

    matrix = np.zeros((len(features), 8), dtype=int)

    for i, feature in enumerate(features):
        matrix[i, 0] = type_ids[i]
        matrix[i, 1] = token_ids[i]
        matrix[i, 2] = feature["depth"]
        matrix[i, 3] = feature["children_count"]
        matrix[i, 4] = int(feature["is_leaf"])
        matrix[i, 5] = feature["token_length"]
        matrix[i, 6] = int(feature["token_is_keyword"])
        matrix[i, 7] = feature["sibling_index"]

    if len(matrix) > max_nodes:
        matrix = matrix[:max_nodes]
    elif len(matrix) < max_nodes:
        pad = np.zeros((max_nodes - len(matrix), matrix.shape[1]), dtype=int)
        matrix = np.vstack((matrix, pad))

    return matrix
