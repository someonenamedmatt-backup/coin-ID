from __future__ import division
import tensorflow as tf
from coin import Coin
SIZE = (64,64)



def read_coin(filename_queue, distort_img, transform = None):
    """Reads and parses coin data
      Args:
    filename_queue: A queue of strings with the filenames to read from.
    distort img (whether to distory the image randomly)
    transform image: None (converts to 64x64)
                    'crp' crops and centers the image
                    'rad' crops and centers the image then transforms radially
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
    result.height = 64
    result.width = 64
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    reader = tf.WholeFileReader()
    result.key, value = reader.read(filename_queue)
    decoded = tf.cast(tf.decode_raw(value, tf.float64),tf.float32)

    result.grade = tf.cast(tf.slice(decoded, [image_bytes],[label_bytes]), tf.int32)
    result.name = tf.cast(tf.slice(decoded, [image_bytes+1],[label_bytes]), tf.int32)
    result.image = tf.image.resize_images(tf.reshape(tf.slice(decoded, [0], [image_bytes]),[result.height,result.width,result.depth]),SIZE[0],SIZE[1])

    if distort_img:
          result.image = tf.image.random_flip_left_right(result.image)
            # Randomly flip the image horizontally.
                # Because these operations are not commutative, consider randomizing
          # the order their operation.
          result.image = tf.image.random_brightness(result.image,
                                                       max_delta=63)
          result.image = tf.image.random_contrast(result.image,
                                                     lower=0.2, upper=1.8)

          # Subtract off the mean and divide by the variance of the pixels.



    result.image = tf.image.per_image_whitening(result.image)

    #not sure which is heigh vs width but its consistent
    return result

def input(file_list, batch_size, distort_img = False, make_coins = False, name_lbls = None, coin_prop = 'img'):
    """Construct a queued batch of images and labels.
    make_coins creates tempory binarized files to represent the coins
    # name_lbls are the names of the coin in the list (if given must be same lenght as file_list)
    """
    # for f in file_list:
    #     if not tf.gfile.Exists(f):
    #         raise ValueError('Failed to find file: ' + f)
    if make_coins:
      for i, f  in enumerate(file_list):
        coin = Coin().make_from_image(f, size = SIZE)
        if name_lbls is not None:
            coin.binarize_coin(f+'_tmp', name_lbl = name_lbls[i], coin_prop = coin_prop)
        else:
            coin.binarize_coin(f+'_tmp', coin_prop = coin_prop)
      file_list = map(lambda name: name + '_tmp', file_list)

    filename_queue = tf.train.string_input_producer(file_list)
    read_input = read_coin(filename_queue, distort_img)
    num_preprocess_threads = 16
    image_batch, grade_batch, name_batch = tf.train.batch(
                                [read_input.image,read_input.grade,read_input.name],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=batch_size * 4)

    # Display the training images in the visualizer.
    tf.image_summary('images', image_batch)
    return image_batch, tf.reshape(grade_batch, [batch_size]), tf.reshape(name_batch, [batch_size])
