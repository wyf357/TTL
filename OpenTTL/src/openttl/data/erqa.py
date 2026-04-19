"""ERQA (Embodied Reasoning Question Answer) dataset loader for TFRecord format.

This module provides utilities to load and parse the ERQA benchmark from TFRecord files.
The dataset contains multimodal interleaved images and text as multiple-choice questions.
"""

from __future__ import annotations

from typing import Iterator, Optional

import tensorflow as tf
from PIL import Image


def parse_erqa_example(example_proto: tf.Tensor) -> dict:
    """Parse a single TFRecord example from the ERQA dataset.
    
    Args:
        example_proto: Serialized TFRecord example
        
    Returns:
        Dictionary containing:
            - question: The text question (str)
            - images: List of PIL Image objects
            - answer: Ground truth answer letter (str)
            - question_type: Type of question (str, optional)
            - visual_indices: Indices for image placement (numpy array)
    """
    feature_description = {
        'question': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'answer': tf.io.FixedLenFeature([], tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
    }
    
    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse tensors to dense
    parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
    parsed_features['question_type'] = tf.sparse.to_dense(parsed_features['question_type'])
    parsed_features['visual_indices'] = tf.sparse.to_dense(parsed_features['visual_indices'])
    
    return parsed_features


def decode_example_to_dict(example: dict) -> dict:
    """Convert TensorFlow tensors to Python objects for easier handling.
    
    Args:
        example: Parsed TFRecord example with tensor values
        
    Returns:
        Dictionary with decoded values:
            - question: Question text (str)
            - images: List of PIL Image objects
            - answer: Answer letter (str)
            - question_type: Question type (str)
            - visual_indices: Numpy array of visual indices
    """
    # Decode question and answer
    question = example['question'].numpy().decode('utf-8')
    answer = example['answer'].numpy().decode('utf-8')
    
    # Decode question type
    question_type_tensors = example['question_type']
    question_type = ""
    if len(question_type_tensors) > 0:
        question_type = question_type_tensors[0].numpy().decode('utf-8')
    
    # Decode images
    images_encoded = example['image/encoded'].numpy()
    images = []
    for img_encoded in images_encoded:
        img_tensor = tf.io.decode_image(img_encoded)
        img_numpy = img_tensor.numpy()
        # Convert numpy array to PIL Image
        if img_numpy.shape[-1] == 1:
            # Grayscale
            img_pil = Image.fromarray(img_numpy.squeeze(), mode='L')
        elif img_numpy.shape[-1] == 3:
            # RGB
            img_pil = Image.fromarray(img_numpy, mode='RGB')
        elif img_numpy.shape[-1] == 4:
            # RGBA
            img_pil = Image.fromarray(img_numpy, mode='RGBA')
        else:
            raise ValueError(f"Unsupported image channels: {img_numpy.shape}")
        images.append(img_pil)
    
    # Get visual indices
    visual_indices = example['visual_indices'].numpy()
    
    return {
        'question': question,
        'images': images,
        'answer': answer,
        'question_type': question_type,
        'visual_indices': visual_indices,
    }


def load_erqa_dataset(
    tfrecord_path: str,
    num_examples: Optional[int] = None,
) -> Iterator[dict]:
    """Load ERQA dataset from TFRecord file.
    
    Args:
        tfrecord_path: Path to the ERQA TFRecord file
        num_examples: Maximum number of examples to load (None for all)
        
    Yields:
        Dictionary for each example with question, images, answer, etc.
    """
    # Create TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse examples
    dataset = dataset.map(parse_erqa_example)
    
    # Limit number of examples if specified
    if num_examples is not None:
        dataset = dataset.take(num_examples)
    
    # Iterate and decode
    for example in dataset:
        yield decode_example_to_dict(example)


def get_erqa_dataset_size(tfrecord_path: str) -> int:
    """Get the total number of examples in the ERQA TFRecord file.
    
    Args:
        tfrecord_path: Path to the ERQA TFRecord file
        
    Returns:
        Number of examples in the dataset
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    count = 0
    for _ in dataset:
        count += 1
    return count
