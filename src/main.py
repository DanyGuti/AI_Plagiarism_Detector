"""
- Main module
Preprocess data, create AST embeddings and train
a CNN model for binary plagiarism detection in code.
Based on research paper:
"Plagiarism Detection in Source Code using Machine Learning."
"""
import numpy as np
import keras

# Uncomment the following lines to enable
# generating pipeline (need datasets to fully run pipeline)
from train import (
    ast_embedding,
    evaluate_saved_model,
    get_project_path,
    load_dataset,
    prepare_model_inputs,
    # preprocess_and_split_ast_data,
    # train_and_evaluate_model,
    # train_dense_ast_random_search,
)
# from preprocessing.lexer_preprocessor import (
#     # process_all_files_lexer,
#     # create_dictionary_from_files_lexer,
#     # create_data_split_from_dict_lexer,
# )
from train_embedding import (
    load_embeddings_and_labels,
    load_label_json,
    padding_embeddings,
    train_full_model,
    build_token_lstm_model,
    evaluate_full_model,
    train_dense_ast_lstm_random_search,
    train_dense_ast_lstm_hyperband,
    plot_best_model,
    init_lstm_model_hb
)

import tensorflow as tf

# Use all 8 logical threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Uncomment the following lines to enable
# data augmentation (need datasets to fully run pipeline)
# from preprocessing.data_augmentor import augment_data_directory

if __name__ == "__main__":
    ############################################################
    # Pipeline for LSTM data generation:
    # source_path = get_project_path("data", "cases")
    # output_path = get_project_path("cases_lexer")
    # process_all_files_lexer(source_path, output_path)
    # input_path = get_project_path("cases_lexer")
    # matrix_dict, original_dict = create_dictionary_from_files_lexer(input_path)
    # print(matrix_dict)
    # create_data_split_from_dict_lexer(matrix_dict, original_dict)
    # print("Data split created successfully from lexer preprocessed files.")

    # Start training model with padding LSTM embeddings
    sequences, token_labels = load_embeddings_and_labels(
        split_dir="train",
        base_dir=get_project_path("embed_data"),
    )
    sequences_val, labels_val = load_embeddings_and_labels(
        split_dir="validation",
        base_dir=get_project_path("embed_data"),
    )
    sequences_test, labels_test = load_embeddings_and_labels(
        split_dir="test",
        base_dir=get_project_path("embed_data"),
    )
    embedded_paddings = padding_embeddings(sequences, max_length=700)
    embedded_paddings_val = padding_embeddings(sequences_val, max_length=700)
    embedded_paddings_test = padding_embeddings(sequences_test, max_length=700)
    # lstm_model = build_token_lstm_model(
    #     vocab_size=500,
    #     embedding_dim=128,
    #     lstm_units=128,
    #     max_length=700
    # )
    # # ############################################################

    # # # Pipeline ran once to prepare the dataset and train the model.
    # # # print("Starting plagiarism detection pipeline...")

    # # # Optional: run these once to prepare the dataset
    # # preprocess_and_split_ast_data()
    # # augment_data_directory()

    train_path = get_project_path("matrix_data", "train")
    val_path = get_project_path("matrix_data", "validation")
    test_path = get_project_path("matrix_data", "test")

    train_features, train_labels = load_dataset(train_path)
    val_features, val_labels = load_dataset(val_path)
    test_features, test_labels = load_dataset(test_path)

    train_inputs = prepare_model_inputs(train_features)
    val_inputs = prepare_model_inputs(val_features)
    test_inputs = prepare_model_inputs(test_features)

    # # # print("Building AST embedding model...")
    ast_model = ast_embedding("ast")
    _ = ast_model(train_inputs)
    train_inputs_dict = {
        "tokens_input": embedded_paddings,
        "ast_type_id": train_features["type_ids"],
        "ast_token_id": train_features["token_ids"],
        "ast_depth": train_features["depth"],
        "ast_children_count": train_features["children_count"],
        "ast_is_leaf": train_features["is_leaf"],
    }
    val_inputs_dict = {
        "tokens_input": embedded_paddings_val,
        "ast_type_id": val_features["type_ids"],
        "ast_token_id": val_features["token_ids"],
        "ast_depth": val_features["depth"],
        "ast_children_count": val_features["children_count"],
        "ast_is_leaf": val_features["is_leaf"],
    }
    # train_full_model(
    #     embedding_model=ast_model,
    #     lstm_model=lstm_model,
    #     labels=train_labels,
    #     input_data=train_inputs_dict,
    #     val_data=(
    #         val_inputs_dict,
    #         val_labels
    #     ),
    #     test_data=test_inputs,
    #     name_model="full_lstm_ast_cnn_modelv2.keras"
    # )

    # print("Training with dense model and random search...")
    # # best_model, best_params = train_dense_ast_lstm_random_search(
    # #     ast_model, lstm_model, train_inputs_dict, train_labels, val_inputs_dict, val_labels, n_trials=10
    # # )
    best_lstm_model, best_lstm_hp, feature_extractor = init_lstm_model_hb(
        lstm_train_inputs=train_inputs_dict["tokens_input"],
        train_labels=token_labels,
        lstm_val_inputs=val_inputs_dict["tokens_input"],
        lstm_val_labels=val_labels
    )
    feature_extractor.trainable = False
    best_model, best_params = train_dense_ast_lstm_hyperband(
        ast_model, feature_extractor, train_inputs_dict, train_labels, val_inputs_dict, val_labels
    )
    best_model.save("best_ast_lstm_dense_hp_v2_model.keras")

    # train_and_evaluate_model()

    # print("Evaluating on test set...")
    # test_ast_path = get_project_path("matrix_data", "test")
    # lstm_path = get_project_path("embed_data")

    # evaluate_full_model(
    #     "best_dense_ast_model_hyperband.keras",
    #     test_path=test_ast_path,
    #     test_lstm_path=lstm_path
    # )
    # # evaluate_saved_model(
    # #     "best_dense_ast_model_hyperband.keras",
    # # )
    # plot_best_model(
    #     labels=train_labels,
    #     input_data=train_inputs_dict,
    #     val_data=(
    #         val_inputs_dict,
    #         val_labels
    #     ),
    #     name_model="best_dense_ast_model_hyperband.keras"
    # )
