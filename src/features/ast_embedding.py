'''
AST embedding for feature extraction from Matrix encoding.
'''
import keras
import tensorflow as tf

# CONSTANTS CAN BE MOVED TO A CONFIG FILE
# CAN BE TUNED BASED ON THE DATASET
MAX_LENGTH_NODES= 350
TYPE_VOCAB_SIZE = 300
TYPE_EMBEDDING_DIM = 64
TOKEN_VOCAB_SIZE = 300
TOKEN_EMBEDDING_DIM = 64

def ast_embedding(
    name_prefix: str,
) -> keras.Model:
    '''
    Generate an embedding for the AST of a given file.
    Args:
        file_path (str): The path to the Java file.
    Returns:
        Model: The embedding of the AST.
    '''
    depth: keras.KerasTensor = keras.Input(
        shape=(MAX_LENGTH_NODES, 1), name=f"{name_prefix}_depth"
    )
    children_count: keras.KerasTensor = keras.Input(
        shape=(MAX_LENGTH_NODES, 1),
        name=f"{name_prefix}_children_count"
    )
    is_leaf: keras.KerasTensor = keras.Input(
        shape=(MAX_LENGTH_NODES, 1),
        name=f"{name_prefix}_is_leaf"
    )
    type_id: keras.KerasTensor = keras.Input(
        shape=(MAX_LENGTH_NODES,),
        name=f"{name_prefix}_type_id"
    )
    token_id: keras.KerasTensor = keras.Input(
        shape=(MAX_LENGTH_NODES,),
        name=f"{name_prefix}_token_id"
    )
    # Embedding for the token and type ids
    type_embedding: tf.Tensor  = keras.layers.Embedding(
        input_dim=TYPE_VOCAB_SIZE,
        output_dim=TYPE_EMBEDDING_DIM,
        name=f"{name_prefix}_type_embedding"
    )(type_id)
    token_embedding: tf.Tensor = keras.layers.Embedding(
        input_dim=TOKEN_VOCAB_SIZE,
        output_dim=TOKEN_EMBEDDING_DIM,
        name=f"{name_prefix}_token_embedding"
    )(token_id)
    # Concatenate the embeddings with the other features to create the final embedding
    # Returning a single tensor dimmension (batch_size, MAX_NODES_LENGTH, 5 + EMBEDDING_DIMS)
    features: tf.Tensor = keras.layers.Concatenate()([
        depth,
        children_count,
        is_leaf,
        type_embedding,
        token_embedding
    ])
    # Following the feature extraction from the paper:
    # Plagiarism Detection in Source Code using Machine Learning.
    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(0.02)
    )(features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(
        padding="same"
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(0.02)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(
        padding="same"
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(0.02)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(
        padding="same"
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    return keras.Model(
        inputs=[depth, children_count, is_leaf, type_id, token_id],
        outputs=x,
        name=f"{name_prefix}_ast_embedding"
    )
