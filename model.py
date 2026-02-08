"""
Model Architecture Module

U-Net with attention gates for vibration mode prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Concatenate, Add, Multiply, BatchNormalization, LeakyReLU
)
from typing import Tuple


def conv_block(x, num_filters: int, kernel_size: Tuple[int, int] = (5, 5)):
    """
    Convolutional block with residual connection.

    Parameters
    ----------
    x : tensor
        Input tensor.
    num_filters : int
        Number of filters.
    kernel_size : tuple, optional
        Kernel size (default: (5, 5)).

    Returns
    -------
    tensor
        Output tensor.
    """
    x_res = x
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x_res = Conv2D(num_filters, (1, 1), padding='same')(x_res)
    x = Add()([x, x_res])
    return x


def attention_gate(input_tensor, gating_tensor, inter_channels: int):
    """
    Attention gate mechanism for skip connections.

    Parameters
    ----------
    input_tensor : tensor
        Input from encoder (skip connection).
    gating_tensor : tensor
        Gating signal from decoder.
    inter_channels : int
        Number of intermediate channels.

    Returns
    -------
    tensor
        Attention-weighted output.
    """
    gating_resized = Conv2D(inter_channels, (1, 1), padding='same')(gating_tensor)
    gating_resized = LeakyReLU(alpha=0.01)(gating_resized)
    input_resized = Conv2D(inter_channels, (1, 1), padding='same')(input_tensor)
    input_resized = LeakyReLU(alpha=0.01)(input_resized)

    combined = Add()([input_resized, gating_resized])
    combined = LeakyReLU(alpha=0.01)(combined)
    attention_weights = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(combined)
    output = Multiply()([input_tensor, attention_weights])
    return output


def decoder_block_with_attention(up_input, skip_input, num_filters: int):
    """
    Decoder block with attention gate.

    Parameters
    ----------
    up_input : tensor
        Input from previous decoder layer.
    skip_input : tensor
        Skip connection from encoder.
    num_filters : int
        Number of filters.

    Returns
    -------
    tensor
        Output tensor.
    """
    upsampled = UpSampling2D((2, 2))(up_input)
    attention_applied = attention_gate(skip_input, upsampled, num_filters // 2)
    concat = Concatenate()([upsampled, attention_applied])
    conv = conv_block(concat, num_filters)
    return conv


def create_unet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    filters: Tuple[int, int, int, int] = (32, 64, 128, 256),
    kernel_size: Tuple[int, int] = (5, 5),
    output_activation: str = 'tanh'
) -> Model:
    """
    Create a U-Net model with attention gates for vibration mode prediction.

    Parameters
    ----------
    input_shape : tuple, optional
        Shape of input images (default: (128, 128, 1)).
    filters : tuple, optional
        Number of filters for each encoder level (default: (32, 64, 128, 256)).
    kernel_size : tuple, optional
        Kernel size for convolutions (default: (5, 5)).
    output_activation : str, optional
        Activation for output layer (default: 'tanh').

    Returns
    -------
    tensorflow.keras.Model
        U-Net model with two inputs: image and k_squared.

    Examples
    --------
    >>> model = create_unet_model()
    >>> model.summary()
    """
    # Main image input
    input_img = Input(shape=input_shape, name='input_image')
    
    # Additional input for k_squared (for physics-informed loss)
    input_k_squared = Input(shape=(1,), name='k_squared')

    # Encoder
    c1 = conv_block(input_img, filters[0], kernel_size)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)
    
    c2 = conv_block(p1, filters[1], kernel_size)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)
    
    c3 = conv_block(p2, filters[2], kernel_size)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Bottleneck
    c5 = conv_block(p3, filters[3], kernel_size)

    # Decoder with Attention Gates
    c7 = decoder_block_with_attention(c5, c3, filters[2])
    c8 = decoder_block_with_attention(c7, c2, filters[1])
    c9 = decoder_block_with_attention(c8, c1, filters[0])

    # Output
    output = Conv2D(1, (1, 1), activation=output_activation, padding='same')(c9)

    # Model with both inputs
    model = Model(inputs=[input_img, input_k_squared], outputs=output)
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get a string representation of the model summary.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The model to summarize.

    Returns
    -------
    str
        Model summary string.
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()
