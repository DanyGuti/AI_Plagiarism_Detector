'''
Main module
Preprocess data, create AST embeddings and train
a CNN model for binary plagiarism detection in code.
Based on research paper:
Plagiarism Detection in Source Code using Machine Learning.
'''
import numpy as np
import os
import keras
from preprocessing.dataset_processor import (
    process_all_files,
    create_dictionary_from_files,
    create_data_split_from_dict,
)
from collections import Counter
from models.cnn_model import (
    load_folder_data,
    prepare_model_inputs,
    binary_plagiarism_code_prediction
)
from features.ast_embedding import (
    ast_embedding
)
from preprocessing.data_augmentor import (
    augment_data_directory
)

CASES: list[str] = [f"case-0{i}" for i in range(1,7)]

def load_all_cases(base_path):
    all_samples = {
        "type_ids": [],
        "token_ids": [],
        "depth": [],
        "children_count": [],
        "is_leaf": [],
    }
    all_labels = []

    for case in CASES:
        case_samples, case_labels = load_folder_data(base_path, case)
        for sample in case_samples:
            # Append each feature array to the corresponding list
            for key in all_samples:
                all_samples[key].append(sample[key])
        all_labels.extend(case_labels)

    # Stack all feature arrays
    for key in all_samples:
        all_samples[key] = keras.preprocessing.sequence.pad_sequences(
            all_samples[key],
            padding='post',
            maxlen=500
        )
    return all_samples, np.array(all_labels)

if __name__ == "__main__":
    print("Main file")
    # Construct the source and destination paths with current working directory
    # Uncomment (Run once to create JSON AST nodes)
    # source_path = os.path.join(os.getcwd(), "..", "data", "cases")
    # output_path = os.path.join(os.getcwd(), "..", "ast_data")
    # process_all_files(source_path, output_path)

    # print("Processing AST matrix data...")

    # input_ast_path = os.path.join(os.getcwd(), "..", "ast_data")

    # # Unpack both returned dictionaries
    # matrix_dict, original_dict = create_dictionary_from_files(input_ast_path)

    # # Pass both dictionaries to the split function
    # create_data_split_from_dict(matrix_dict, original_dict)

    # print("Matrix generation and dataset split completed.")

    # Load the dataset to train the model
    # train_base_path = os.path.join(os.getcwd(), "..", "matrix_data", "train")
    # validation_base_path = os.path.join(os.getcwd(), "..", "matrix_data", "validation")
    # test_base_path = os.path.join(os.getcwd(), "..", "matrix_data", "test")
    # # Load all cases for training
    # case_features, case_labels = load_all_cases(train_base_path)
    # print("Training labels shape:", case_labels.shape)
    # # Load all cases for validation
    # case_features_val, case_labels_val = load_all_cases(validation_base_path)
    # # Load all cases for testing
    # case_features_test, case_labels_test = load_all_cases(test_base_path)

    # # Extract the features
    # inputs = prepare_model_inputs(case_features, name_prefix="ast")
    # val_inputs = prepare_model_inputs(case_features_val, name_prefix="ast")
    # ast_embedding_model = ast_embedding("ast")
    # # pass the inputs to the model
    # ast_output = ast_embedding_model(inputs)

    # model = binary_plagiarism_code_prediction(
    #     ast_embedding_model,
    #     case_labels,
    #     input_data=inputs,
    #     val_data=(val_inputs, case_labels_val)
    # )

    # test_inputs = prepare_model_inputs(case_features_test, name_prefix="ast")
    # results = model.evaluate(
    #     x=test_inputs,
    #     y=case_labels_test,
    # )
    # print(f"Test results: {results}")
    
    #############################
    # PROCESS DATA AUGMENTATION #
    #############################
    augment_data_directory(
       src_csv_path="/Users/Guty/Downloads/conplag/versions"
    )
