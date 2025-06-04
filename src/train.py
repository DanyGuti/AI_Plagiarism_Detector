'''
- Train Module
This module contains functions to preprocess AST data, train a dense
model for binary plagiarism detection,
and evaluate the model's performance. It includes functions for loading datasets,
building models, and plotting training history.
'''
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from preprocessing.dataset_processor import (
    process_all_files,
    create_dictionary_from_files,
    create_data_split_from_dict,
)
from features.ast_embedding import ast_embedding
from models.cnn_model import (
    load_folder_data,
    prepare_model_inputs,
    binary_plagiarism_code_prediction,
)

# File to train the model
# Constant representing the cases to be processed
CASES = [f"case-0{i}" for i in range(1, 29)]


def get_project_path(*parts: tuple[str,str]) -> str:
    '''
    Constructs a path relative to the project root directory.
    Args:
        *parts: Variable length argument list of path components.
    Returns:
        str: The constructed path.
    '''
    return os.path.join(os.getcwd(), "..", *parts)


def preprocess_and_split_ast_data():
    '''
    Preprocesses the AST data by generating JSON nodes and matrix data,
    and then splits the data into training, validation, and test sets.
    '''
    print("Generating AST JSON nodes and matrix data...")
    source_path = get_project_path("data", "cases")
    output_path = get_project_path("ast_data")
    process_all_files(source_path, output_path)
    matrix_dict, original_dict = create_dictionary_from_files(output_path)
    create_data_split_from_dict(matrix_dict, original_dict)
    print("Matrix generation and dataset split completed.")


def load_dataset(base_path: str) -> tuple[dict[str, list], np.array]:
    '''
    Loads the dataset from the specified base path.
    Args:
        base_path (str): The base path where the dataset is located.
    Returns:
        tuple: A tuple containing:
        - all_samples (dict): A dictionary with keys "type_ids", "token_ids",
                "depth", "children_count", "is_leaf",
            each containing a list of corresponding values.
        - all_labels (np.array): An array of labels for the samples.
    '''
    all_samples = {k: [] for k in ["type_ids", "token_ids", "depth", "children_count", "is_leaf"]}
    all_labels = []

    for case in CASES:
        case_samples, case_labels = load_folder_data(base_path, case)[:2]
        for sample in case_samples:
            for key in all_samples:
                all_samples[key].append(sample[key])
        all_labels.extend(case_labels)

    for key in all_samples:
        all_samples[key] = keras.preprocessing.sequence.pad_sequences(
            all_samples[key], padding="post", maxlen=700
        )

    return all_samples, np.array(all_labels)


def build_dense_ast_model(
    embedding_model: keras.Model,
    params: dict[str, int]
) -> keras.Model:
    '''
    Builds a dense model for AST embeddings with specified parameters.
    Args:
        embedding_model (keras.Model): The pre-trained AST embedding model.
        params (dict): A dictionary containing hyperparameters for the model.
    Returns:
        keras.Model: The compiled dense model.
    '''
    x = keras.layers.GlobalAveragePooling1D()(embedding_model.output)
    for i in range(1, 4):
        x = keras.layers.Dense(
            params[f'dense{i}'], kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(params['l2']))(x)
        x = keras.layers.Activation('relu')(x)
        if i < 3:
            x = keras.layers.Dropout(params[f'dropout{i}'])(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=embedding_model.input, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_dense_ast_random_search(
    embedding_model: keras.Model,
    train_inputs: dict[str, list],
    train_labels: np.array,
    val_inputs: dict[str, list],
    val_labels: np.array,
    n_trials=10
) -> tuple[keras.Model, dict]:
    '''
    Trains a dense model using random search for hyperparameter optimization.
    Args:
        embedding_model (keras.Model): The pre-trained AST embedding model.
        train_inputs (dict): The training inputs for the model.
        train_labels (np.array): The labels for the training data.
        val_inputs (dict): The validation inputs for the model.
        val_labels (np.array): The labels for the validation data.
        n_trials (int): The number of trials for random search.
    Returns:
        tuple: A tuple containing:
        - best_model (keras.Model): The best model found during the search.
        - best_params (dict): The hyperparameters of the best model.
    '''
    best_model, best_val_acc, best_params = None, 0.0, None
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(train_labels), y=train_labels
    )
    cw = {i: w for i, w in enumerate(class_weights)}

    for trial in range(n_trials):
        params = {
            'dense1': random.choice([64, 128, 256]),
            'dense2': random.choice([32, 64, 128]),
            'dense3': random.choice([16, 32, 64]),
            'dropout1': random.choice([0.2, 0.3, 0.4]),
            'dropout2': random.choice([0.2, 0.3, 0.4]),
            'l2': random.choice([1e-4, 5e-4, 1e-3]),
            'lr': random.choice([1e-4, 5e-4, 1e-3])
        }
        print(f"\n🔎 Trial {trial+1}/{n_trials} | Params: {params}")
        model = build_dense_ast_model(embedding_model, params)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            train_inputs, train_labels, validation_data=(val_inputs, val_labels),
            epochs=50, batch_size=32, class_weight=cw, callbacks=[early_stop], verbose=2
        )

        val_acc = max(history.history["val_accuracy"])
        print(f"Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_model, best_params = val_acc, model, params
            plot_history(history, save=True, prefix=f"best_model_trial_{trial+1}")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}\n Best Hyperparameters: {best_params}")
    best_model.save("best_dense_ast_model.keras")
    return best_model, best_params


def plot_history(
    history: keras.callbacks.History,
    save=False,
    prefix="plot"
) -> None:
    '''
    Plots the training history of a model.
    Args:
        history (keras.callbacks.History): The training history of the model.
        save (bool): Whether to save the plots to files.
        prefix (str): Prefix for the saved plot filenames.
    Returns:
        None
    '''
    os.makedirs("images", exist_ok=True)
    for metric in ["accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric], label=f'Train {metric.title()}', linestyle='-')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.title()}', linestyle='--')
        plt.title(metric.title())
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        if save:
            plt.savefig(f"images/{prefix}_{metric}.png")
        plt.close()

def evaluate_saved_model(
    model_path: str
) -> None:
    '''
    Evaluates a saved model on the test dataset and prints metrics.
    Args:
        model_path (str): The path to the saved model.
    Returns:
        None
    '''
    test_path = get_project_path("matrix_data", "test")
    features, labels = load_dataset(test_path)
    inputs = prepare_model_inputs(features)
    model = keras.models.load_model(model_path)
    preds = (model.predict(inputs) > 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Print metrics
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Plagiarized", "Plagiarized"],
                yticklabels=["Non-Plagiarized", "Plagiarized"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/confusion_matrix.png")
    plt.show()
    plt.close()

def train_and_evaluate_model() -> None:
    '''
    Trains a CNN model for binary plagiarism detection using AST embeddings,
    evaluates it on the test set, and saves the model.
    Returns:
        None
    '''
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

    model.save("ast_cnn_model.keras")

    results = model.evaluate(x=test_inputs, y=test_labels)
    print(f"Test results: {results}")

    print("Train label distribution:", Counter(train_labels))
    print("Validation label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))

    cm = confusion_matrix(test_labels, (model.predict(test_inputs) > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Plagiarized", "Plagiarized"],
                yticklabels=["Non-Plagiarized", "Plagiarized"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/confusion_matrix_trained.png")
    plt.show()
    plt.close()
