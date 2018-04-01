import random

from PIL import Image

MNIST_PATH = '/Users/kostya/mnist/train'

IMG_SIZE = 28


class DigitClass:
    def __init__(self, digit):
        self.digit = digit
        self.data = [
            [0] * IMG_SIZE for _ in range(IMG_SIZE)
        ]

    def load(self):
        pass

    def _normalize(self, value, max_, min_):
        if value > 0:
            return 255
        elif value < 0:
            return -255
        return 0


        if value == 0:
            return 0
        if value > 0:
            return 255-min(int((value / max_) * 255 / 1.5), 255)
        return -255 - max(-int((value / min_) * 255 / 1.5), -255)

    def normalize(self):
        max_data = max(max(d) for d in self.data)
        min_data = min(min(d) for d in self.data)
        self.data = [
            [self._normalize(d, max_data, min_data) for d in data_row]
            for data_row in self.data
        ]

    def _g(self, value):
        return value if value > 0 else 0

    def _b(self, value):
        return -value if value < 0 else 0

    def save(self):
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        img_data = [
            (0, self._g(self.data[y][x]), self._b(self.data[y][x]))
                for x in range(IMG_SIZE)
                for y in range(IMG_SIZE)
        ]
        img.putdata(img_data)
        img.save(f'./results/{self.digit}.png')

    def train(self, img_path, correct):
        raw_img = read_image(img_path)
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                if correct:
                    self.data[y][x] += (raw_img[y, x] != 255) * 10
                elif self.data[x][y] != 20000:
                    self.data[x][y] -= (raw_img[x, y] != 255)

    def test(self, img_path, cool=False):
        score = 0

        if cool:
            img = Image.open(img_path)
            img.save('./results/cool.png')

        raw_img = read_image(img_path)

        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                if raw_img[y, x]:
                    score += ((255-raw_img[y, x]) * self.data[y][x]) > 20

        return score

    def __str__(self):
        return str(self.digit)


def read_image(path):
    img = Image.open(path)
    return img.load()


def img_path(digit, i=None):
    if i is None:
        i = random.randint(0, 100)

    return f'{MNIST_PATH}/{digit}/{i}.png'


def test(digits):
    percents = []

    for test_digit in range(10):
        results = []

        TEST_COUNT = 40

        for _ in range(TEST_COUNT):
            scores = []

            for digit_cLass in digits:
                score = digit_cLass.test(img_path(test_digit))
                scores.append(score)

            # print(scores)
            results.append(scores.index(max(scores)))

        percents.append(
            (len([x for x in results if x == test_digit]) / TEST_COUNT))
        print(percents[-1])

    print()
    print(sum(percents) / len(percents))


def main():
    digits = [
        DigitClass(i) for i in range(0, 10)
    ]

    # TRAIN
    for digit_cLass in digits:
        for i in range(200):
            digit_cLass.train(
                f'{MNIST_PATH}/{digit_cLass.digit}/{i}.png', True)

        for k in range(10):
            if k != digit_cLass.digit:
                for i in range(200):
                    digit_cLass.train(
                        f'{MNIST_PATH}/{k}/{i}.png', False)

        digit_cLass.normalize()
        digit_cLass.save()

    # TEST

    test(digits)


if __name__ == '__main__':
    main()
