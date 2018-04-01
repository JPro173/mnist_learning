import random
from collections import Counter

from PIL import Image

MNIST_PATH = './test_data/'


class Perceptron:
    def __init__(self, digit, SIZE):
        # mark the perceptron with the related digit
        self.digit = digit

        # two dimensional array for weights
        self.weights = [[0] * SIZE for _ in range(SIZE)]
        self.size = SIZE

        # max value for a weight, can be [-max_value, max_value]
        self.max_value = 100
        # not used, but should be in a proper perceptron
        # for some reason lowers the accuracy if used
        self.threshold = 300000

    def test(self, inputs):
        result = 0

        for x in range(self.size):
            for y in range(self.size):
                # sum all pixels multiplied by corresponding weights
                result += self.weights[y][x] * (255 - inputs[y, x])
        # expected to return boolean value, by comparing to threshold
        # like this: return result > self.threshold
        # but it is hard to find proper threshold for every digit, so just
        # return sum of all (weights[j] * inputs[j])
        return int(result)

    def train(self, inputs, expected):
        # correct answers will be 10 times less, than incorrect =>
        # weight for correct answer is 10 times bigger
        modifier = .01 if expected else -.001

        for x in range(self.size):
            for y in range(self.size):
                self.weights[y][x] += (255 - inputs[y, x]) * modifier

    def normalize(self):
        # find the biggest weight
        max_weight = max([
            max(a) for a in self.weights])

        # FIXME: check also for the smallest weight, and
        # normalize negative values by the smallest

        for x in range(self.size):
            for y in range(self.size):
                # change the weights, so they are in range [-100, 100]
                self.weights[y][x] = (self.weights[y][x] / max_weight) * 100
                self.weights[y][x] = round(self.weights[y][x], 2)


def load_img(digit, i=None):
    if i is None:
        i = random.randint(0, 1000)

    path = f'{MNIST_PATH}/{digit}/{i}.png'

    img = Image.open(path)
    return img.load()


for current_digit in range(10):
    p = Perceptron(current_digit, 28)

    for digit in range(10):
        # number of train subjects for each digit
        # really affects the speed of the script
        for _ in range(200):
            img_data = load_img(digit)
            p.train(img_data, p.digit == digit)

    p.normalize()

    results = Counter()

    for test_digit in range(10):
        # number of tests
        for _ in range(10):
            results[test_digit] += p.test(load_img(test_digit))

    # uncomment this to see scores for all digits
    # print('Number:', current_digit, ' Scores:', results)
    print('Number:', current_digit, ' Guess:', results.most_common(1)[0][0])

