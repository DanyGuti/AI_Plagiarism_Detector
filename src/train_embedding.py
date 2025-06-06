'''
- Full model build Module
'''
from pathlib import Path
import json
import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

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
        64,
        activation='relu',
        name='dense_layer'
    )(x)
    return keras.Model(
        inputs=tokens_input,
        outputs=x,
        name='token_lstm_model'
    )

def train_full_model(
    embedding_model: keras.Model,
    lstm_model: keras.Model,
    labels: np.ndarray,
    input_data: dict[str, np.ndarray],
    val_data: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    test_data: dict[str, np.ndarray],
):
    x_ast = keras.layers.GlobalAveragePooling1D()(embedding_model.output)
    x_lstm = lstm_model.output
    x = keras.layers.Concatenate()([x_ast, x_lstm])
    x = keras.layers.Dense(
        256,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.0002)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(
        64,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.0002)
    )(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(
        16,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0.0002)
    )(x)
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
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
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
    model.save("full_lstm_ast_cnn_model.keras")
    plot_history(history, "full_model_training_history")

    return model
