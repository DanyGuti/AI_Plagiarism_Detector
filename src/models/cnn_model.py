"""
AST CNN model
Takes AST embedding feature extraction from AST nodes
and creates a CNN model for binary classification.
"""

from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight


def load_folder_data(folder_path: str, case_folder: str) -> tuple[list[dict], list[int]]:
    """Load .txt AST matrix files into feature arrays and labels."""
    samples, labels = [], []
    full_path = Path(folder_path) / case_folder

    for file in full_path.glob("*.txt"):
        with file.open("r") as f:
            lines = f.readlines()

        label = 0 if "non" in file.name.lower() and "plagiarized" in file.name.lower() else 1
        sample = {
            "type_ids": [],
            "token_ids": [],
            "depth": [],
            "children_count": [],
            "is_leaf": []
        }

        for line in lines:
            t_id, tok_id, d, ch, leaf = map(int, line.strip().split())
            sample["type_ids"].append(t_id)
            sample["token_ids"].append(tok_id)
            sample["depth"].append(d)
            sample["children_count"].append(ch)
            sample["is_leaf"].append(leaf)

        # Convert to numpy arrays
        for key in sample:
            sample[key] = np.array(sample[key])

        samples.append(sample)
        labels.append(label)

    return samples, labels


def prepare_model_inputs(features: dict[str, np.ndarray], name_prefix="ast") -> dict[str, np.ndarray]:
    """Reshape features into model input tensors."""
    return {
        f"{name_prefix}_type_id": features["type_ids"][..., np.newaxis],
        f"{name_prefix}_token_id": features["token_ids"][..., np.newaxis],
        f"{name_prefix}_depth": features["depth"][..., np.newaxis],
        f"{name_prefix}_children_count": features["children_count"][..., np.newaxis],
        f"{name_prefix}_is_leaf": features["is_leaf"][..., np.newaxis]
    }


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score
from tensorflow import keras
import numpy as np

def binary_plagiarism_code_prediction(
    embedding_model: keras.Model,
    labels: np.ndarray,
    input_data: dict[str, np.ndarray],
    val_data: tuple[dict[str, np.ndarray], np.ndarray],
    test_data: tuple[dict[str, np.ndarray], np.ndarray] = None
) -> keras.Model:
    """Create, train, and evaluate a binary classification CNN model with class weighting and batch normalization."""

    x = keras.layers.GlobalAveragePooling1D()(embedding_model.output)
    x = keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(32, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=embedding_model.input, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "ast_cnn_model.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    history = model.fit(
        x=input_data,
        y=labels,
        epochs=50,
        validation_data=val_data,
        batch_size=32,
        callbacks=[model_checkpoint],
        class_weight=class_weight_dict
    )

    model.save("ast_cnn_model.keras")
    plot_history(history, save=True, prefix="ast_cnn_model")

    if test_data:
        test_loss, test_acc = model.evaluate(test_data[0], test_data[1])
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    return model


def plot_history(history: keras.callbacks.History, save: bool = False, prefix: str = "training_plot"):
    """Plot and optionally save training history."""
    image_dir = "images"
    if save:
        os.makedirs(image_dir, exist_ok=True)

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='-')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save:
        plt.savefig(os.path.join(image_dir, f"{prefix}_accuracy.png"))
    else:
        plt.show()
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss', linestyle='-')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save:
        plt.savefig(os.path.join(image_dir, f"{prefix}_loss.png"))
    else:
        plt.show()
    plt.close()
