from __future__ import division
import tensorflow as tf
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
    result.height = 128
    result.width = 128
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

def input(file_list, batch_size, distort_img = False):
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

SEARCH_MIN = 1.0
SEARCH_MAX = 500.0
MIN_RADIUS = 50

def _find_circle(img):
    #Takes the image and tries to find circles in it
    #plays around with sensitivity until exactly one circle is found
    #Colors everything outside of that circle white
    #returns false if we don't find a circle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100, param2 = SEARCH_MIN, minRadius = MIN_RADIUS)
    if circles is None:
        return False
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100, param2 = SEARCH_MAX, minRadius = MIN_RADIUS)
    g_1 = SEARCH_MAX
    g_0 = SEARCH_MIN
    counter = 0
    while circles is None or len(circles[0])>1:
        if counter >= 20:
            raise False
        param2 = (g_1+g_0)/2
        circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100, param2 = param2, minRadius = MIN_RADIUS)
        if circles is None:
            # guess higher
            g_1 =  param2
        else:
            #guess loawer
            g_0 = param2
        counter += 1
    x, y, r = np.round(circles[0,0]).astype(int)
    cv2.circle(img, center = (x, y), radius = r+120, color= (255, 255, 255), thickness = 240)
    return img

def _resize_image(img, size = (128,128)):
    #takes an image (length x width x 3 with rgb colors)
    #and a tuple of size in pixels
    #returns the image normalized in hsv coloring
    #which cuts out whitespace and centers the image
    #returns false if something doesn't work
    if img.shape[-1] != 3:
        return False
    hsv = colors.rgb_to_hsv(img)
    w = np.nonzero((hsv[:,:,2] < 160) & (hsv[:,:,1] < 160))
    min_x, min_y = np.min(w,axis=1)
    max_x, max_y = np.max(w,axis=1)
    ims = misc.imresize(img[min_x:max_x,min_y:max_y,:],size)/256.0
    return ims

def _convert_to_radian(img):
    #transforms the image radially
    max=(2.0, img.shape[1]/(2*math.pi), 1)
    mid=(img.shape[0]/2.0,img.shape[1]/2.0,0)
    def polar_to_euclidean(pos):
        pos = (pos[0]/max[0],pos[1]/max[1],pos[2])
        nPos = (pos[0]*math.cos(pos[1]) + mid[0], pos[0]*math.sin(pos[1]) + mid[1],pos[2])
        return nPos
    rImg = scipy.ndimage.interpolation.geometric_transform(img,polar_to_euclidean, img.shape)
    return rImg
