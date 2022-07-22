
import colorsys
from itertools import product

from PIL import Image

from .base import BaseCaptcha


class DemoCaptcha(BaseCaptcha):
    white_chars = '02468BDFHJLNPRTVXZ'
    charset = set(white_chars)

    @classmethod
    @property
    def tesseract_config_array(cls):
        config = super().tesseract_config_array
        config = [
            *config,
            '-c', f'tessedit_char_whitelist={cls.white_chars}',
        ]
        return config

    def __init__(self):
        super().__init__()
        self.tesseract_lang = 'demo'
        self.image_ext = '.gif'

    def preprocess(self, image_fp):
        image = super().preprocess(image_fp)
        width = image.width
        height = image.height
        index = lambda x, y: y * width + x

        # Binaryzation.
        image = image.convert('RGB')
        bin_data = [1] * (width * height)
        max_l = 0.6
        min_s = 0.65
        for (y, x), (r, g, b) in zip(product(range(height), range(width)), image.getdata()):
            _, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            if 0 < x < width - 1 and 0 < y < height - 1 and l <= max_l and s >= min_s:
                bin_data[index(x, y)] = 0

        # Remove random dots.
        # Left, right, up, down.
        straight_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Up-right, up-left, down-left, down-right.
        diagonal_offsets = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
        clean_data = list(bin_data)
        for y, x in product(range(1, height - 1), range(1, width - 1)):
            center_index = index(x, y)
            center_value = bin_data[center_index]
            straight_sum = sum(bin_data[index(x + h, y + v)] for h, v in straight_offsets)
            diagonal_sum = sum(bin_data[index(x + h, y + v)] for h, v in diagonal_offsets)
            if center_value == 1:
                if straight_sum == 0 and diagonal_sum <= 1:
                    clean_data[center_index] = 0
            elif center_value == 0:
                if straight_sum == 4 or (diagonal_sum == 4 and straight_sum >= 2):
                    clean_data[center_index] = 1

        image = Image.new('1', image.size)
        image.putdata(clean_data)

        return image

    def validate_captcha_input(self, captcha):
        if len(captcha) != 5:
            raise ValueError('number of characters is not 5')

        invalid_chars = set(captcha) - self.charset
        if invalid_chars:
            raise ValueError(f'contains invalid chars {invalid_chars}')
