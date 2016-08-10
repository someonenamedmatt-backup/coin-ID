from __future__ import division
def read_coin(filename_queue):
    """Reads and parses coin data
      Args:
    filename_queue: A queue of strings with the filenames to read from.
    Returns:
        An object representing a single example, with the following fields:
        height: number of rows in the result (128)
        width: number of columns in the result (128)
        depth: number of color channels in the result (3)
        label: an int32 Tensor with the label
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
    decoded = tf.decode_raw(value, tf.float64)
    print decoded.get_shape()
    result.label = tf.cast(tf.slice(decoded, [image_bytes],[label_bytes]), tf.int32)
    result.image = tf.reshape(tf.slice(decoded, [0], [image_bytes]),[result.height,result.width,result.depth])
    #not sure which is heigh vs width but its consistent
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
    image: 3-D Tensor of [height, width, 3] of type.float64.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                                        [image, label],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 3 * batch_size,
                                        min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size=batch_size,
                                    num_threads=num_preprocess_threads,
                                    capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
#     tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, file_list, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """

    for f in file_list:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(file_list)
    read_input = read_coin(filename_queue)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(read_input.image, read_input.label,
                                        batch_size * 3, batch_size, shuffle = False)
                
