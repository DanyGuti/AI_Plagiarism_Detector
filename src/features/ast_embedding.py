'''
- AST Embedding Model
This module defines a Keras model for AST (Abstract Syntax Tree) feature
extraction with Embedding layers.
'''

import keras
import tensorflow as tf

# Constants for AST embedding model
MAX_LENGTH_NODES = 700
TYPE_VOCAB_SIZE = 300
TYPE_EMBEDDING_DIM = 128
TOKEN_VOCAB_SIZE = 300
TOKEN_EMBEDDING_DIM = 128

def ast_embedding(
    name_prefix: str,
) -> keras.Model:
    '''
    Create an AST embedding model using Keras.
    Args:
        name_prefix (str): Prefix for the input names in the model.
    Returns:
        keras.Model: A Keras model for AST feature extraction.
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

    type_embedding_layer: tf.Tensor = keras.layers.Embedding(
        input_dim=TYPE_VOCAB_SIZE,
        output_dim=TYPE_EMBEDDING_DIM,
        name=f"{name_prefix}_type_embedding"
    )(type_id)
    token_embedding_layer: tf.Tensor = keras.layers.Embedding(
        input_dim=TOKEN_VOCAB_SIZE,
        output_dim=TOKEN_EMBEDDING_DIM,
        name=f"{name_prefix}_token_embedding"
    )(token_id)

    features: tf.Tensor = keras.layers.Concatenate()([
        depth,
        children_count,
        is_leaf,
        type_embedding_layer,
        token_embedding_layer
    ])

    l2_reg = 0.002 # Increased L2 regularization factor

    conv_branch_a = keras.layers.Conv1D(
        filters=64, # Reduced filters
        kernel_size=3,
        padding="same",
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(features)
    conv_branch_a = keras.layers.BatchNormalization()(conv_branch_a)
    conv_branch_a = keras.layers.Activation("relu")(conv_branch_a)

    conv_branch_b = keras.layers.Conv1D(
        filters=48, # Reduced filters
        kernel_size=5,
        padding="same",
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(features)
    conv_branch_b = keras.layers.BatchNormalization()(conv_branch_b)
    conv_branch_b = keras.layers.Activation("relu")(conv_branch_b)

    conv_branch_c = keras.layers.Conv1D(
        filters=32, # Reduced filters
        kernel_size=7,
        padding="same",
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(features)
    conv_branch_c = keras.layers.BatchNormalization()(conv_branch_c)
    conv_branch_c = keras.layers.Activation("relu")(conv_branch_c)

    x = keras.layers.Concatenate()([conv_branch_a, conv_branch_b, conv_branch_c])
    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Dropout(0.45)(x) # Increased dropout

    x = keras.layers.Conv1D(
        filters=96, # Reduced filters
        kernel_size=5,
        padding="same",
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Dropout(0.45)(x) # Increased dropout


    return keras.Model(
        inputs=[depth, children_count, is_leaf, type_id, token_id],
        outputs=x,
        name=f"{name_prefix}_ast_embedding"
    )
