'''Main module for the project'''
# from preprocessing import process_all_files
from features.ast_embedding import traverse_ast,\
    read_ast_from_file, encode_features
import os

if __name__=="__main__":
    print("Main file")
    # Construct the source and destination paths with current working directory
    # source_path = os.path.join(os.getcwd(), "..", "data", "cases")
    # output_path = os.path.join(os.getcwd(), "..", "ast_data")
    # process_all_files(source_path, output_path)
    tree = read_ast_from_file(
        os.path.join(
            os.getcwd(),
            "..",
            "ast_data",
            "case-01",
            "non-plagiarized",
            "01",
            "T01.json"
        ))
    graph_tree = traverse_ast(tree)
    print(encode_features(graph_tree))
