import numpy as np
from scipy import misc
from PIL import Image
import pickle
import cv2

IMAGE_SIZE = 28


def images_to_sprite(data):
    """
    Creates the sprite image

    Parameters
    ----------
        data: [batch_size, height, weight, n_channel]

    Returns
    -------
      data: Sprited image::[height, weight, n_channel]
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


if __name__ == '__main__':
    data = []
    with open('/home/pham.hoang.anh/prj/face_detect/X_train_triplet.pkl', 'rb') as f:
        X = pickle.load(f)
    for x in X:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)        
            img = misc.imresize(x, (IMAGE_SIZE, IMAGE_SIZE, 3))
            data.append(img)
    img_sprite = images_to_sprite(np.array(data))
    sprite = Image.fromarray(img_sprite.astype(np.uint8))
    sprite.save("/home/pham.hoang.anh/prj/face_detect/visualize/128D-Facenet-LFW-Embedding-Visualisation/oss_data/LFW_HA_sprites.png")
