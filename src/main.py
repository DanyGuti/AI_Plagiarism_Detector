import os
from preprocessing.dataset_processor import (
    process_all_files,
    create_dictionary_from_files,
    create_data_split_from_dict,
)

if __name__ == "__main__":
    print("Main file")
    # Construct the source and destination paths with current working directory 
    # Uncomment (Run once to create JSON AST nodes)
    # source_path = os.path.join(os.getcwd(), "..", "data", "cases")
    # output_path = os.path.join(os.getcwd(), "..", "ast_data")
    # process_all_files(source_path, output_path)

    print("Processing AST matrix data...")

    input_ast_path = os.path.join(os.getcwd(), "..", "ast_data")

    # Unpack both returned dictionaries
    matrix_dict, original_dict = create_dictionary_from_files(input_ast_path)

    # Pass both dictionaries to the split function
    create_data_split_from_dict(matrix_dict, original_dict)

    print("Matrix generation and dataset split completed.")
