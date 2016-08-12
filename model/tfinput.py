from __future__ import division
import tensorflow as tf

def read_coin(filename_queue):
    """Reads and parses coin data
      Args:
    filename_queue: A queue of strings with the filenames to read from.
    Returns:
        An object representing a single example, with the following fields:
        height: number of rows in the result (128)
        width: number of columns in the result (128)
        depth: number of color channels in the result (3)
        grade: an int32 Tensor with the condition of the coin label
        name: an int32 Tensor with the name of the coin label
        uint8image: a [height, width, depth] float64 Tensor with the image data
     """
    class ImageRecord(object):
        pass

    result = ImageRecord()

    label_bytes = 1
    result.height = 128
    result.width = 128
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    reader = tf.WholeFileReader()
    result.key, value = reader.read(filename_queue)
    decoded = tf.cast(tf.decode_raw(value, tf.float64),tf.float32)

    result.grade = tf.cast(tf.slice(decoded, [image_bytes],[label_bytes]), tf.int32)
    result.name = tf.cast(tf.slice(decoded, [image_bytes+1],[label_bytes]), tf.int32)
    result.image = tf.reshape(tf.slice(decoded, [0], [image_bytes]),[result.height,result.width,result.depth])
    #not sure which is heigh vs width but its consistent
    return result

def input(file_list, batch_size):
    """Construct a queued batch of images and labels.
    Args:
    image: 3-D Tensor of [height, width, 3] of type.float64.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # for f in file_list:
    #     if not tf.gfile.Exists(f):
    #         raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(file_list)
    read_input = read_coin(filename_queue)
    num_preprocess_threads = 16
    image_batch, grade_batch, name_batch = tf.train.batch(
                                [read_input.image,read_input.grade,read_input.name],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=batch_size * 4)

    # Display the training images in the visualizer.
    # tf.image_summary('images', image_batch)
    return image_batch, tf.reshape(grade_batch, [batch_size]), tf.reshape(name_batch, [batch_size])
