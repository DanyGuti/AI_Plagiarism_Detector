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
    # TODO:
        # MISSING PADDING THE DATA WITH KERAS.PREPROCESSING.PADSEQUENCES
    full_path: Path = Path(folder_path) / Path(case_folder)
    label = None
    for file in full_path.glob("*.txt"):
        type_ids, token_ids, depth, children_count, is_leaf =\
            [], [], [], [], [], []
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
            labels.append(label)
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
    inputs = {
        f"{name_prefix}_depth": features["depth"][..., np.newaxis],
        f"{name_prefix}_children_count": features["children_count"][..., np.newaxis],
        f"{name_prefix}_is_leaf": features["is_leaf"][..., np.newaxis],
        f"{name_prefix}_type_id": features["type_ids"][..., np.newaxis],
        f"{name_prefix}_token_id": features["token_ids"][..., np.newaxis],
    }
    return inputs

def binary_plagiarism_code_prediction(
    embeding_model: keras.Input,
    labels: np.ndarray,
    input_data: dict[str, keras.KerasTensor],
    val_data: dict[str, keras.KerasTensor],
) -> None:
    inputs = embeding_model.inputs
    ast_output = embeding_model.outputs[0]
    x = keras.layers.GlobalMaxPool1D()(ast_output)
    output = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'accuracy'],
    )
    model.summary()
    model.save("ast_cnn_model.keras")

    history = model.fit(
        x=input_data,
        y=labels,
        epochs=10,
        validation_data=val_data,
    )
    plot_history(
        history,
    )
    return model

def plot_history(
    history: keras.callbacks.History,
) -> None:
    '''
    Plot the training and validation accuracy and loss.
    Args:
        history (keras.callbacks.History): The training history.
        save_path (str): The path to save the plot.
    '''
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy')   # blue line
    plt.plot(history.history['val_accuracy'], 'bo', label='Val Accuracy') # blue circle markers only
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()