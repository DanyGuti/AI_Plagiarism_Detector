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
    test_data: tuple[dict[str, keras.KerasTensor], np.ndarray] = None,
) -> keras.Model:
    inputs = embedding_model.inputs
    ast_output = embedding_model.outputs[0]
    
    x = keras.layers.Flatten()(ast_output)  # Flatten the output of the embedding model
    x = keras.layers.Dropout(0.3)(x)  # Regularization
    x = keras.layers.Dense(64, activation='relu')(x)  # Smaller dense layer
    x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # early_stop = keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=3,
    #     restore_best_weights=True
    # )

    history = model.fit(
        x=input_data,
        y=labels,
        epochs=50,
        validation_data=val_data,
        # callbacks=[early_stop],
        batch_size=16  # Helps with small data
    )

    model.save("ast_cnn_model.keras")
    plot_history(history, save=True, prefix="ast_cnn_model")

    if test_data is not None:
        test_loss, test_accuracy = model.evaluate(test_data[0], test_data[1])
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return model


import matplotlib.pyplot as plt
from tensorflow import keras
import os
# ...existing code...
def plot_history(history: keras.callbacks.History, save: bool = False, prefix: str = "training_plot") -> None:
    '''
    Plot the training and validation accuracy and loss.

    Args:
        history (keras.callbacks.History): The training history.
        save (bool): Whether to save the plots as PNG files instead of showing them.
        prefix (str): Filename prefix if saving plots.
    '''
    image_dir = "images"
    if save:
        os.makedirs(image_dir, exist_ok=True)

    # Plot training & validation accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'bo', label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    if save:
        plt.savefig(os.path.join(image_dir, f'{prefix}_accuracy.png'))
    else:
        plt.show()
    plt.close()

    # Plot training & validation loss
    plt.figure()
    plt.plot(history.history['loss'], 'r-', label='Train Loss')
    plt.plot(history.history['val_loss'], 'ro', label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    if save:
        plt.savefig(os.path.join(image_dir, f'{prefix}_loss.png'))
    else:
        plt.show()
    plt.close()
