"""
Main module
Preprocess data, create AST embeddings and train
a CNN model for binary plagiarism detection in code.
Based on research paper:
Plagiarism Detection in Source Code using Machine Learning.
"""

import os
import numpy as np
import keras
from preprocessing.dataset_processor import (
    process_all_files,
    create_dictionary_from_files,
    create_data_split_from_dict,
)
from models.cnn_model import (
    load_folder_data,
    prepare_model_inputs,
    binary_plagiarism_code_prediction,
)
from features.ast_embedding import ast_embedding
from preprocessing.data_augmentor import augment_data_directory
from collections import Counter


CASES: list[str] = [f"case-0{i}" for i in range(1, 29)]


def get_project_path(*parts) -> str:
    return os.path.join(os.getcwd(), "..", *parts)


def preprocess_and_split_ast_data():
    print("Generating AST JSON nodes and matrix data...")
    source_path = get_project_path("data", "cases")
    output_path = get_project_path("ast_data")
    process_all_files(source_path, output_path)

    matrix_dict, original_dict = create_dictionary_from_files(output_path)
    create_data_split_from_dict(matrix_dict, original_dict)
    print("Matrix generation and dataset split completed.")


def load_dataset(base_path: str):
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
            for key in all_samples:
                all_samples[key].append(sample[key])
        all_labels.extend(case_labels)

    for key in all_samples:
        all_samples[key] = keras.preprocessing.sequence.pad_sequences(
            all_samples[key], padding="post", maxlen=500
        )

    return all_samples, np.array(all_labels)


def train_and_evaluate_model():
    train_path = get_project_path("matrix_data", "train")
    val_path = get_project_path("matrix_data", "validation")
    test_path = get_project_path("matrix_data", "test")

    train_features, train_labels = load_dataset(train_path)
    val_features, val_labels = load_dataset(val_path)
    test_features, test_labels = load_dataset(test_path)

    train_inputs = prepare_model_inputs(train_features, name_prefix="ast")
    val_inputs = prepare_model_inputs(val_features, name_prefix="ast")
    test_inputs = prepare_model_inputs(test_features, name_prefix="ast")

    ast_model = ast_embedding("ast")
    _ = ast_model(train_inputs)  # Run once to initialize

    model = binary_plagiarism_code_prediction(
        ast_model,
        train_labels,
        input_data=train_inputs,
        val_data=(val_inputs, val_labels),
        test_data=(test_inputs, test_labels),
    )

    results = model.evaluate(x=test_inputs, y=test_labels)
    print(f"Test results: {results}")

    print("Train label distribution:", Counter(train_labels))
    print("Validation label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))



def augment_code_data():
    augment_data_directory(src_csv_path="conplag/versions")


if __name__ == "__main__":
    print("Starting plagiarism detection pipeline...")

    # UNCOMMENT to run once for preprocessing:
    # preprocess_and_split_ast_data()

    # UNCOMMENT to run data augmentation:
    # augment_code_data()

    train_and_evaluate_model()
