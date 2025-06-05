"""
- Main module
Preprocess data, create AST embeddings and train
a CNN model for binary plagiarism detection in code.
Based on research paper:
"Plagiarism Detection in Source Code using Machine Learning."
"""

# Uncomment the following lines to enable
# generating pipeline (need datasets to fully run pipeline)
from train import (
    # ast_embedding,
    evaluate_saved_model,
    # get_project_path,
    # load_dataset,
    # prepare_model_inputs,
    # preprocess_and_split_ast_data,
    # train_and_evaluate_model,
    # train_dense_ast_random_search,
)

# Uncomment the following lines to enable
# data augmentation (need datasets to fully run pipeline)
# from preprocessing.data_augmentor import augment_data_directory

if __name__ == "__main__":
    # Pipeline ran once to prepare the dataset and train the model.
    print("Starting plagiarism detection pipeline...")

    # Optional: run these once to prepare the dataset
    # preprocess_and_split_ast_data()
    # augment_data_directory()

    # train_path = get_project_path("matrix_data", "train")
    # val_path = get_project_path("matrix_data", "validation")
    # test_path = get_project_path("matrix_data", "test")

    # train_features, train_labels = load_dataset(train_path)
    # val_features, val_labels = load_dataset(val_path)
    # test_features, test_labels = load_dataset(test_path)

    # train_inputs = prepare_model_inputs(train_features)
    # val_inputs = prepare_model_inputs(val_features)
    # test_inputs = prepare_model_inputs(test_features)

    # print("Building AST embedding model...")
    # ast_model = ast_embedding("ast")
    # _ = ast_model(train_inputs)

    # print("Training with dense model and random search...")
    # best_model, best_params = train_dense_ast_random_search(
    #     ast_model, train_inputs, train_labels, val_inputs, val_labels, n_trials=10
    # )
    # best_model.save("best_ast_dense_model.keras")

    # train_and_evaluate_model()

    print("Evaluating on test set...")
    evaluate_saved_model("best_ast_dense_modelV4.keras")
