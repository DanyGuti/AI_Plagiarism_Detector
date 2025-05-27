'''
AST CNN model 
Takes AST embedding feature extraction from AST nodes
and creates a CNN model for binary classification.
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import keras

def load_folder_data(
    folder_path: str,
    case_folder: str = None,
) -> np.ndarray:
    '''
    Load the data from a folder and assign a label to it.
    Args:
        folder_path (str): The path to the folder.
    Returns:
        np.array: The data and labels.
    '''
    # Load the data from the folder
    samples = []
    labels = []
    # Loop through the directory
    # Read the TXT matrices files, extract features by columns
    # and append them to the lists
    full_path: Path = Path(folder_path) / Path(case_folder)
    label = None
    for file in full_path.glob("*.txt"):
        type_ids, token_ids, depth, children_count, is_leaf =\
            [], [], [], [], []
        with file.open("r") as f:
            lines = f.readlines()
        label = 0 if "nonplag" in file.name.lower() else 1
        for line in lines:
            columns = line.strip().split(' ')
            type_ids.append(int(columns[0]))
            token_ids.append(int(columns[1]))
            depth.append(int(columns[2]))
            children_count.append(int(columns[3]))
            is_leaf.append(int(columns[4]))
        sample_features = {
            "type_ids": np.array(type_ids),
            "token_ids": np.array(token_ids),
            "depth": np.array(depth),
            "children_count": np.array(children_count),
            "is_leaf": np.array(is_leaf),
        }
        samples.append(sample_features)
        labels.append(label)

    return samples, labels

def prepare_model_inputs(features, name_prefix="ast"):
    print("features['depth'] =", features["depth"], type(features["depth"]))
    inputs = {
        f"{name_prefix}_depth": np.array(features["depth"])[..., np.newaxis],
        f"{name_prefix}_children_count": np.array(features["children_count"])[..., np.newaxis],
        f"{name_prefix}_is_leaf": np.array(features["is_leaf"])[..., np.newaxis],
        f"{name_prefix}_type_id": np.array(features["type_ids"])[..., np.newaxis],
        f"{name_prefix}_token_id": np.array(features["token_ids"])[..., np.newaxis],
    }
    return inputs

from tensorflow import keras
import numpy as np

def binary_plagiarism_code_prediction(
    embedding_model: keras.Model,
    labels: np.ndarray,
    input_data: dict[str, keras.KerasTensor],
    val_data: tuple[dict[str, keras.KerasTensor], np.ndarray],
) -> keras.Model:
    inputs = embedding_model.inputs
    ast_output = embedding_model.outputs[0]
    
    x = keras.layers.GlobalAveragePooling1D()(ast_output)
    x = keras.layers.Dropout(0.3)(x)  # Regularization
    x = keras.layers.Dense(64, activation='relu')(x)  # Smaller dense layer
    x = keras.layers.Dropout(0.3)(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x=input_data,
        y=labels,
        epochs=50,
        validation_data=val_data,
        callbacks=[early_stop],
        batch_size=16  # Helps with small data
    )

    model.save("ast_cnn_model.keras")
    plot_history(history)
    return model


def plot_history(history: keras.callbacks.History) -> None:
    '''
    Plot the training and validation accuracy and loss.
    Args:
        history (keras.callbacks.History): The training history.
    '''
    # Plot training & validation accuracy values
    plt.plot(history.history['binary_accuracy'], 'b-', label='Train Accuracy')   # blue line
    plt.plot(history.history['val_binary_accuracy'], 'bo', label='Val Accuracy') # blue circle markers
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], 'r-', label='Train Loss')
    plt.plot(history.history['val_loss'], 'ro', label='Val Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()