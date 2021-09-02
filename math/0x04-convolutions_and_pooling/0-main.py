#!/usr/bin/env python3

# import matplotlib.pyplot as plt
# import numpy as np
# convolve_channels = __import__('4-convolve_channels').convolve_channels


# if __name__ == '__main__':

#     dataset = np.load('../../data/animals_1.npz')
#     images = dataset['data']
#     print(images.shape)
#     kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1],
#                       [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
#     images_conv = convolve_channels(images, kernel, padding='valid')
#     print(images_conv.shape)

#     plt.imshow(images[0])
#     plt.show()
#     plt.imshow(images_conv[0])
#     plt.show()


# import numpy as np
# convolve_channels = __import__('4-convolve_channels').convolve_channels

# np.random.seed(4)
# m = np.random.randint(100, 200)
# h, w = np.random.randint(20, 50, 2).tolist()
# c = np.random.randint(2, 5)
# fh, fw = (np.random.randint(2, 5, 2)).tolist()
# sh, sw = (np.random.randint(2, 4, 2)).tolist()

# images = np.random.randint(0, 256, (m, h, w, c))
# kernel = np.random.randint(0, 10, (fh, fw, c))
# conv_ims = convolve_channels(images, kernel, stride=(sh, sw))
# np.set_printoptions(threshold=np.inf)
# print(conv_ims)
# print(conv_ims.shape)


# import matplotlib.pyplot as plt
# import numpy as np
# convolve = __import__('5-convolve').convolve


# if __name__ == '__main__':

#     dataset = np.load('../../data/animals_1.npz')
#     images = dataset['data']
#     print(images.shape)
#     kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
#                        [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0],
#                                                                [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
#                        [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])

#     images_conv = convolve(images, kernels, padding='valid')
#     print(images_conv.shape)

#     plt.imshow(images[0])
#     plt.show()
#     plt.imshow(images_conv[0, :, :, 0])
#     plt.show()
#     plt.imshow(images_conv[0, :, :, 1])
#     plt.show()
#     plt.imshow(images_conv[0, :, :, 2])
#     plt.show()


import numpy as np
convolve = __import__('5-convolve').convolve

np.random.seed(5)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
cin = np.random.randint(2, 5)
cout = np.random.randint(5, 10)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w, cin))
kernel = np.random.randint(0, 10, (fh, fw, cin, cout))
conv_ims = convolve(images, kernel, stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(conv_ims)
print(conv_ims.shape)