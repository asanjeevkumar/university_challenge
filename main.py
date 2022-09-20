# This is a sample Python script.
import argparse

import numpy
from sklearn.datasets import load_digits
from scipy import stats


def mode_of_images(img1: numpy.ndarray, img2: numpy.ndarray):
    diff_in_image = img1 - img2
    return stats.mode(diff_in_image)[0]


def function_1(img1, img2):
    mode_of = mode_of_images(img1, img2)
    return mode_of.max(axis=0).max()


def function_2(img1, img2):
    mode_of = mode_of_images(img1, img2)
    count_x, count_y = img1.shape[0], img2.shape[0]
    sum_of_x = mode_of.sum(axis=0)
    inverse_x_and_sum_of_x = (1/count_x)*sum_of_x
    result = (1/count_y)*inverse_x_and_sum_of_x.sum()
    return result


def function_3(img1, img2):
    mode_of = mode_of_images(img1, img2)
    count_y = img2.shape[0]
    max_x = mode_of.max(axis=0)
    return (1/count_y) * max_x.sum()


def function_4(img1, img2):
    mode_of = mode_of_images(img1, img2)
    count_x = img1.shape[0]
    sum_of_x = mode_of.sum(axis=0)
    inverse_x_and_sum_of_x = (1/count_x) * sum_of_x
    return inverse_x_and_sum_of_x.max()


def main(image_index_1, image_index_2):
    digits = load_digits(return_X_y=False)
    print(f"Return of function 1: {function_1(digits.images[image_index_1], digits.images[image_index_2])}")
    print(f"Return of function 2: {function_2(digits.images[image_index_1], digits.images[image_index_2])}")
    print(f"Return of function 3: {function_3(digits.images[image_index_1], digits.images[image_index_2])}")
    print(f"Return of function 4: {function_4(digits.images[image_index_1], digits.images[image_index_2])}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_1", type=int, help="Index of first image tobe used for process. Should be int ")
    parser.add_argument("image_2", type=int, help="Index of second image tobe used for  process. Should be int ")

    args = parser.parse_args()
    main(args.image_1, args.image_2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
