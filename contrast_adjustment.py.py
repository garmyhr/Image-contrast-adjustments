import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import floor, ceil
import sys

def get_hist(img, G=2**8):

    m, n = np.shape(img)
    h = np.zeros(G)
    p = np.zeros(G)
    c = np.zeros(G)

    # Counting grey tones
    for value in img.flatten():
        h[int(value)] += 1

    # Creating normalized histogram
    for i in range(0, G):
        p[i] = h[i] / (n*m)

    # Creating cumulative histogram
    c[0] = p[0]
    for i in range(1, G):
        c[i] = c[i-1] + p[i]

    return h, p, c

def create_transformation_matrix(h, window, G=2**8):

    c = np.zeros(G)
    c[0] = h[0]
    for i in range(1, G):
        c[i] = c[i-1] + h[i]
    c = c / window

    T = np.zeros(G)
    for i in range(G):
        T[i] = round((G-1) * c[i])

    return T

def histogram_equalization(img, G=2**8):

    m, n = np.shape(img)
    h = np.zeros(G)
    T = np.zeros(G)
    processed_img = np.zeros((m,n))


    for value in img.ravel():
        h[int(value)] += 1

    T = create_transformation_matrix(h, (m*n))

    for x in range(0, m):
        for y in range(0, n):
            processed_img[x, y] = T[int(img[x, y])]


    plt.subplot(2,3,1)
    plt.title("Histogram before")
    plt.hist(img.flatten(), G, [0,G])

    plt.subplot(2,3,2)
    plt.title("Cumulative histogram before")
    plt.hist(img.flatten(), G, [0,G], density=True, cumulative=True)

    plt.subplot(2,3,3)
    plt.title("Picture before")
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)

    plt.subplot(2,3,4)
    plt.title("Histogram after")
    plt.hist(processed_img.flatten(), G, [0,G])

    plt.subplot(2,3,5)
    plt.title("Cumulative histogram after")
    plt.hist(processed_img.flatten(), G, [0,G], density=True, cumulative=True)

    plt.subplot(2,3,6)
    plt.title("Picture after")
    plt.imshow(processed_img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def adaptive_histogram_equalization(img, G=2**8, window_size=3):

    pad_size = int((window_size-1)/2)
    img = np.pad(img, (pad_size, pad_size), mode='constant')
    m, n = np.shape(img)
    processed_img = np.zeros((m,n))

    for x in range(pad_size, m - pad_size):

        print("%d of %d" % (x, m-pad_size-1))
        h = np.zeros(G)
        window = img[x - pad_size : x + pad_size+1, pad_size-pad_size : pad_size + pad_size+1]

        for value in window.ravel():
            i = int(value)
            h[i] += 1

        T = create_transformation_matrix(h, (window_size*window_size))

        # Transform pixel in the first window
        processed_img[x, pad_size] = T[int(img[x, pad_size])]

        for y in range(pad_size, n - pad_size):

            left_col = img[x - pad_size : x + pad_size+1, y - pad_size-1]
            right_col = img[x - pad_size : x + pad_size+1, y + pad_size]

            for pixel in left_col:
                g1 = int(pixel)
                h[g1] -= 1
            for pixel in right_col:
                g1 = int(pixel)
                h[g1] += 1

            T = create_transformation_matrix(h, (window_size*window_size))

            # transformer piksel i sentrum av vindauget
            processed_img[x, y] = T[int(img[x, y])]

    plt.figure()
    plt.title("Adaptive histogram equalization")
    plt.imshow(processed_img, cmap="gray", vmin=0, vmax=255)
    plt.show()


def clahe(img, clip_value=10, G=2**8, window_size=3):

    pad_size = int((window_size-1)/2)
    img = np.pad(img, (pad_size, pad_size), mode='constant')
    m, n = np.shape(img)
    processed_img = np.zeros((m,n))
    left_col = np.zeros(window_size)
    right_col = np.zeros(window_size)

    for x in range(pad_size, m - pad_size):

        print("%d of %d" % (x, m-pad_size-1))
        h = np.zeros(G)
        window = img[x - pad_size : x + pad_size+1, pad_size-pad_size : pad_size + pad_size+1]

        for value in window.ravel():
            i = int(value)
            h[i] += 1

        clipped_values = [min(val, clip_value) for val in h]
        to_add = (np.sum(h) - np.sum(clipped_values)) / G
        for element in clipped_values:
            element += to_add

        T = create_transformation_matrix(clipped_values, (window_size*window_size))

        # Transform pixel in the first window
        processed_img[x, pad_size] = T[int(img[x, pad_size])]

        for y in range(pad_size+1, n - pad_size):

            left_col = img[x - pad_size : x + pad_size+1, y - pad_size-1] # -1 for left column of previous window
            right_col = img[x - pad_size : x + pad_size+1, y + pad_size] # right column of current window

            for pixel in left_col:
                g1 = int(pixel)
                h[g1] -= 1
            for pixel in right_col:
                g1 = int(pixel)
                h[g1] += 1

            clipped_values = [min(val, clip_value) for val in h]
            to_add = (np.sum(h) - np.sum(clipped_values)) / G
            for element in clipped_values:
                element += to_add

            T = create_transformation_matrix(clipped_values, (window_size*window_size))
            # Transform pixel in window center
            processed_img[x, y] = T[int(img[x, y])]

    plt.figure()
    plt.title("CLAHE")
    plt.imshow(processed_img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def main():
    img = cv2.imread(sys.argv[1], 0)
    histogram_equalization(img)
    adaptive_histogram_equalization(img, window_size=21)
    clahe(img, clip_value=20, window_size=21)

if __name__ == "__main__":
    main()