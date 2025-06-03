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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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

    train_inputs = prepare_model_inputs(train_features)
    val_inputs = prepare_model_inputs(val_features)
    test_inputs = prepare_model_inputs(test_features)

    ast_model = ast_embedding("ast")
    _ = ast_model(train_inputs)

    model = binary_plagiarism_code_prediction(
        ast_model,
        train_labels,
        input_data=train_inputs,
        val_data=(val_inputs, val_labels),
        test_data=(test_inputs, test_labels),
    )

    model.save("ast_cnn_model.keras")  # Save model

    results = model.evaluate(x=test_inputs, y=test_labels)
    print(f"Test results: {results}")

    print("Train label distribution:", Counter(train_labels))
    print("Validation label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))


def evaluate_saved_model(model_path: str):
    """
    Load a saved model, evaluate it on test data, print metrics, and save confusion matrix.
    """
    test_path = get_project_path("matrix_data", "test")
    test_features, test_labels = load_dataset(test_path)
    test_inputs = prepare_model_inputs(test_features)

    model = keras.models.load_model(model_path)
    preds = model.predict(test_inputs)
    preds_binary = (preds > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(test_labels, preds_binary, target_names=["Non-Plagiarized", "Plagiarized"]))

    cm = confusion_matrix(test_labels, preds_binary)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Plagiarized", "Plagiarized"],
        yticklabels=["Non-Plagiarized", "Plagiarized"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/confusion_matrix.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Starting plagiarism detection pipeline...")

    # UNCOMMENT to run once for preprocessing:
    # preprocess_and_split_ast_data()

    # UNCOMMENT to run data augmentation:
    # augment_code_data()

    # UNCOMMENT to train and save model
    # train_and_evaluate_model()

    evaluate_saved_model(model_path="ast_cnn_modelV2.keras")
