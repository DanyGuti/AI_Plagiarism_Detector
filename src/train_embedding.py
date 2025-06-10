'''
- Full model build Module
'''
from pathlib import Path
import json
import os
import keras
import keras_tuner as kt
import random
import uuid
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score, f1_score,\
    confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from train import get_project_path, load_dataset, prepare_model_inputs
from models.cnn_model import plot_history

CASES = [f"case-0{i}" for i in range(1, 29)]

def load_embeddings_and_labels(
    split_dir: Path,
    base_dir: Path,
) -> tuple[list, np.ndarray]:
    sequences = []
    labels = []

    split_dir = Path(Path(base_dir) / Path(split_dir))
    for case in CASES:
        full_path = Path(split_dir) / Path(case)
        for file in full_path.glob("*.txt"):
            with file.open("r") as f:
                lines = f.readlines()
            vectors = [
                np.array(
                    list(map(float, line.strip().split()))
                )
                for line in lines
            ]
            sample_embedding = np.vstack(vectors)
            sequences.append(sample_embedding)
            label = 0 if "non" in file.name.lower() and "plagiarized" in file.name.lower() else 1
            labels.append(label)
    return sequences, np.array(labels)

def load_label_json(label_json_path: Path) -> dict:
    label_data_split = {}
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data_split = json.load(f)
    return label_data_split


def padding_embeddings(
    sequences: list,
    max_length: int = 700
) -> np.ndarray:
    padded_sequences = keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    return padded_sequences

def build_token_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    max_length: int = 700
) -> keras.Model:
    tokens_input = keras.Input(
        shape=(max_length, 10),name='tokens_input'
    )
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            name='lstm_layer',
            recurrent_dropout=0.2,
            dropout=0.2,
        )
    )(tokens_input)
    x = keras.layers.Dense(
        128,
        name='dense_layer',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.002),
        activation='relu'
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(
        64,
        name='dense_layer_2',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.002),
        activation='relu'
    )(x)
    x = keras.layers.Dense(
        32,
        name='dense_layer_3',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.002),
        activation='relu'
    )(x)
    return keras.Model(
        inputs=tokens_input,
        outputs=x,
        name='token_lstm_model'
    )

def init_lstm_model_hb(
    lstm_train_inputs: keras.Input,
    train_labels: np.ndarray,
    lstm_val_inputs: keras.Input,
    lstm_val_labels: np.ndarray,
    max_epochs: int = 30
):
    def build_lstm(hp):
        model, _ = build_token_lstm_model_hb(hp)
        return model
    tuner = kt.Hyperband(
        build_lstm,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3,
        directory='hyperband_results',
        project_name='token_lstm_tuning'
    )
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10
    )
    tuner.search(
        lstm_train_inputs,
        train_labels,
        validation_data=(lstm_val_inputs, lstm_val_labels),
        epochs=max_epochs,
        batch_size=32,
        callbacks=[stop_early],
        verbose=2
    )
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model, feature_extractor = build_token_lstm_model_hb(best_hp)
    return best_model, best_hp.values, feature_extractor

def build_token_lstm_model_hb(
    hp: kt.HyperParameters,
    lstm_units: int = 128,
    max_length: int = 700
) -> keras.Model:
    unique_id = str(uuid.uuid4())
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    tokens_input = keras.Input(
        shape=(max_length, 10),name='tokens_input'
    )
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            kernel_initializer='he_normal',
            name=f"lstm_layer_{unique_id}",
            recurrent_dropout=dropout_rate,
            dropout=dropout_rate,
        )
    )(tokens_input)

    for i in range(1, 4):
        x = keras.layers.Dense(
            hp.Int(f'dense{i}', min_value=16, max_value=256, step=16),
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(hp.Float('l2', 1e-5, 1e-2, sampling='log')),
            name=f"dense_{i}_{unique_id}"
        )(x)
        x = keras.layers.BatchNormalization(name=f"batch_norm_{i}_{unique_id}")(x)
        x = keras.layers.Activation('relu', name=f"relu_{i}_{unique_id}")(x)
        if i < 3:
            x = keras.layers.Dropout(hp.Float(f'dropout{i}', 0.2, 0.5, step=0.1), name=f"dropout_{i}_{unique_id}")(x)
    feature_extractor = keras.Model(
        inputs=tokens_input,
        outputs=x,
        name=f'token_lstm_feature_extractor_{unique_id}'
    )
    output = keras.layers.Dense(1, activation="sigmoid", name=f"output_{unique_id}")(x)
    model = keras.Model(
        inputs=tokens_input,
        outputs=output,
        name=f'token_lstm_model_hb_{unique_id}'
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('lr', 1e-5, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, feature_extractor

def train_full_model(
    embedding_model: keras.Model,
    lstm_model: keras.Model,
    labels: np.ndarray,
    input_data: dict[str, np.ndarray],
    val_data: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    test_data: dict[str, np.ndarray],
    name_model: str,
):
    x_ast = keras.layers.GlobalAveragePooling1D()(embedding_model.output)
    x_lstm = lstm_model.output
    x = keras.layers.Concatenate()([x_ast, x_lstm])
    x = keras.layers.Dense(
        16,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.00098943)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(
        128,
        kernel_initializer='he_normal',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(
        48,
        kernel_initializer='he_normal',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    for i, input_tensor in enumerate(embedding_model.inputs):
        print(f"Input {i}: {input_tensor.name}, shape={input_tensor.shape}")
    model = keras.Model(
        inputs={
            "tokens_input": lstm_model.input,
            "ast_type_id": embedding_model.input[3],
            "ast_token_id": embedding_model.input[4],
            "ast_depth": embedding_model.input[0],
            "ast_children_count": embedding_model.input[1],
            "ast_is_leaf": embedding_model.input[2],
        },
        outputs=output,
        name="full_model"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00036381),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "full_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
    ]
    print(lstm_model.input.shape)

    history = model.fit(
        x=input_data,
        y=labels,
        validation_data=val_data,
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    model.save(f"{name_model}.keras")
    plot_history(history, save=True, prefix="training_full_model_lstm_ast_attention")

    return model

def evaluate_full_model(
    model_path: str,
    test_path: str,
    test_lstm_path: str,
):
    test_sequences, test_labels = load_embeddings_and_labels(
        split_dir="test",
        base_dir=test_lstm_path,
    )
    test_padded = padding_embeddings(test_sequences, max_length=700)
    test_features, test_labels = load_dataset(test_path)
    test_features = prepare_model_inputs(test_features)

    test_inputs = {
        "tokens_input": test_padded,
        **test_features,
    }
    print("Inspecting test input shapes:")
    for key, value in test_inputs.items():
        print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
    # Load the best model
    model: keras.Model = keras.models.load_model(model_path)
    model.summary()
    preds = (model.predict(test_inputs) > 0.55).astype(int)

    # Compute metrics
    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot CM
    cm = confusion_matrix(test_labels, preds)

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

    loss, accuracy = model.evaluate(test_inputs, test_labels)

    print("Plotting the test accuracy and loss:")
    plt.figure(figsize=(12, 6))
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


def build_dense_ast_lstm_model(
    embedding_model: keras.Model,
    lstm_model: keras.Model,
    params: dict[str, int]
) -> keras.Model:
    '''
    Builds a dense model for AST embeddings and LSTM embeddings with specified parameters.
    Args:
        embedding_model (keras.Model): The pre-trained AST embedding model.
        lstm_model (keras.Model): The pre-trained LSTM model for token embeddings.
        params (dict): A dictionary containing hyperparameters for the model.
    Returns:
        keras.Model: The compiled dense model.
    '''
    x_ast = keras.layers.GlobalAveragePooling1D()(embedding_model.output)
    x_lstm = lstm_model.output
    x = keras.layers.Concatenate()([x_ast, x_lstm])
    for i in range(1, 4):
        x = keras.layers.Dense(
            params[f'dense{i}'], kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(params['l2']))(x)
        x = keras.layers.Activation('relu')(x)
        if i < 3:
            x = keras.layers.Dropout(params[f'dropout{i}'])(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs={
        "tokens_input": lstm_model.input,
        "ast_type_id": embedding_model.input[3],
        "ast_token_id": embedding_model.input[4],
        "ast_depth": embedding_model.input[0],
        "ast_children_count": embedding_model.input[1],
        "ast_is_leaf": embedding_model.input[2],
    },
    outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_dense_ast_lstm_random_search(
    embedding_model: keras.Model,
    lstm_model: keras.Model,
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
        lstm_model (keras.Model): The pre-trained LSTM model for token embeddings.
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
        model = build_dense_ast_lstm_model(embedding_model, lstm_model, params)
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

def build_dense_ast_lstm_model_tuner(
    hp: kt.HyperParameters,
    embedding_model: keras.Model,
    lstm_model: keras.Model
) -> keras.Model:
    '''
    Builds a dense model for AST embeddings and LSTM embeddings with hyperparameters from Keras Tuner.
    Args:
        hp (kt.HyperParameters): The hyperparameters from Keras Tuner.
        embedding_model (keras.Model): The pre-trained AST embedding model.
        lstm_model (keras.Model): The pre-trained LSTM model for token embeddings.
    Returns:
        keras.Model: The compiled dense model.
    '''
    x_ast = keras.layers.GlobalAveragePooling1D(name="global_avg_pool_ast")(embedding_model.output)
    x_lstm = lstm_model.output
    x = keras.layers.Concatenate(name="concat_dense_ast_lstm")([x_ast, x_lstm])
    unique_id = str(uuid.uuid4())
    for i in range(1, 4):
        x = keras.layers.Dense(
            hp.Int(f'dense{i}', min_value=16, max_value=256, step=16),
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(hp.Float('l2', 1e-5, 1e-2, sampling='log')),
            name=f"dense_{i}_{unique_id}"
        )(x)
        x = keras.layers.BatchNormalization(name=f"batch_norm_{i}_{unique_id}")(x)
        x = keras.layers.Activation('relu', name=f"relu_{i}")(x)
        if i < 3:
            x = keras.layers.Dropout(hp.Float(f'dropout{i}', 0.2, 0.5, step=0.1), name=f"dropout_{i}_{unique_id}")(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs={
            "tokens_input": lstm_model.input,
            "ast_type_id": embedding_model.input[3],
            "ast_token_id": embedding_model.input[4],
            "ast_depth": embedding_model.input[0],
            "ast_children_count": embedding_model.input[1],
            "ast_is_leaf": embedding_model.input[2],
        }, outputs=output
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('lr', 1e-5, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_dense_ast_lstm_hyperband(
    embedding_model: keras.Model,
    lstm_model: keras.Model,
    train_inputs: dict[str, list],
    train_labels: np.array,
    val_inputs: dict[str, list],
    val_labels: np.array,
    max_epochs=50,
) -> tuple[keras.Model, dict]:
    '''
    Trains a dense model using Hyperband for hyperparameter optimization.
    Args:
        embedding_model (keras.Model): The pre-trained AST embedding model.
        lstm_model (keras.Model): The pre-trained LSTM model for token embeddings.
        train_inputs (dict): The training inputs for the model.
        train_labels (np.array): The labels for the training data.
        val_inputs (dict): The validation inputs for the model.
        val_labels (np.array): The labels for the validation data.
        max_epochs (int): The maximum number of epochs to train.
        factor (int): The reduction factor for Hyperband.
    Returns:
        tuple: A tuple containing:
        - best_model (keras.Model): The best model found during the search.
        - best_params (dict): The hyperparameters of the best model.
    '''
    # Placeholder for Hyperband implementation
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(train_labels), y=train_labels
    )
    cw = {i: w for i, w in enumerate(class_weights)}
    def build_model(hp):
        return build_dense_ast_lstm_model_tuner(hp, embedding_model, lstm_model)
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3,
        directory='hyperband_results_v2',
        project_name='dense_ast_lstm_tuning'
    )
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10
    )

    tuner.search(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=max_epochs,
        batch_size=32,
        class_weight=cw,
        callbacks=[stop_early],
        verbose=2
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    best_model.fit(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=max_epochs,
        batch_size=32,
        class_weight=cw,
        verbose=2
    )

    best_model.save("best_dense_ast_model_v2_hyperband.keras")
    return best_model, best_hp.values


def plot_best_model(
    labels: np.ndarray,
    input_data: dict[str, np.ndarray],
    val_data: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    name_model: str
):
    model = keras.models.load_model(name_model)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "full_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
    ]
    history = model.fit(
        x=input_data,
        y=labels,
        validation_data=val_data,
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    plot_history(history, prefix="best_model_hp_tuner")
