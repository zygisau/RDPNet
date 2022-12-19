import tensorflow as tf

data_dir = '/Users/zygimantas/Documents/Dataset/ChangeDetectionDataset/Real/subset/train/OUT/'
img_height, img_width = 256, 256
batch_size = 2

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  labels=None)


def count_neighbours(image, ncolours=5, adjacency=1):
    """
    Note colours have to be numbers 1 to ncolours, NOT zero
    Zero values are created during the calc, both in padding and in tensor products,
    and must be discarded
    """
    f = 2 * adjacency + 1      # filter size
    f = tf.cast(f, tf.int32)
    ic = adjacency * (f + 1)   # Index of central pixel in flattened patch
    ic = tf.cast(ic, tf.int32)
    print(f"{adjacency=} {ic=} {f=}")
    print(f"{tf.shape(image)=}")

    # Get patches. Returns location of patch in dims 1, 2, and (flattened) patch in 3
    patches = tf.image.extract_patches(image, sizes=(1, f, f, 1), strides=(1, 1, 1, 1),
                                       padding='SAME', rates=[1, 1, 1, 1])
    print(f"{patches.shape=}")
    patches_for_one_hot = tf.cast(patches, tf.int32)

    # A sort of outer product with one-hot encoding of the central pixel in a patch
    # So if a patch [i, j, k, :] has central pixel l, it is copied to [i, j, k, :, l]
    # First, create a one-hot encoding of the central pixel in each patch
    # one-hot of central pixel in each patch
    oh_central = tf.one_hot(
        patches_for_one_hot[:, :, :, ic], axis=-1, depth=ncolours + 1, dtype=tf.float32)
    print(f"{oh_central.shape=}")

    # Want oh1[m, h, w, ipixel, colour] = patches[m, h, w, ipixel] * oh_central[m, h, w, colour]
    oh1 = tf.einsum('ijkl,ijkm->ijklm', patches, oh_central)
    oh1 = tf.cast(oh1, tf.int32)
    print(f"{oh1.shape=}")

    # One-hot encode the patches
    oh2 = tf.one_hot(oh1, axis=-1, depth=ncolours + 1)
    print(f"{oh2.shape=}")

    # Set central pixels to zero, since a pixel is not counted as being adjacent to itself
    mask = tf.concat([
        tf.ones((oh2.shape[0], oh2.shape[1], oh2.shape[2], oh2.shape[3] //
                2, oh2.shape[4], oh2.shape[5]), dtype=tf.int32),
        tf.zeros((oh2.shape[0], oh2.shape[1], oh2.shape[2],
                 1, oh2.shape[4], oh2.shape[5]), dtype=tf.int32),
        tf.ones((oh2.shape[0], oh2.shape[1], oh2.shape[2], oh2.shape[3] //
                2, oh2.shape[4], oh2.shape[5]), dtype=tf.int32),
    ], axis=3)
    mask = tf.cast(mask, tf.float32)
    assert mask.shape == oh2.shape
    oh3 = oh2 * mask

    # finally, reduce sum, discard counts of zeros and return result
    return tf.reduce_sum(oh3, axis=[1, 2, 3])[:, 1:, 1:]


# train_ds = train_ds.take(1).map(lambda x: count_neighbours(x, 2))
for image in train_ds.as_numpy_iterator():
    result = count_neighbours(image, 2)
    print(f"{result}")
