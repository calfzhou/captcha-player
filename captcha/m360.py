
from datetime import datetime
import string
import time

from .base import BaseCaptcha


class M360Captcha(BaseCaptcha):
    white_chars = f'{string.digits}{string.ascii_lowercase}'
    black_chars = '01loy'
    charset = set(white_chars) - set(black_chars)

    @classmethod
    @property
    def tesseract_config_array(cls):
        config = super().tesseract_config_array
        config = [
            *config,
            '-c', f'tessedit_char_whitelist={cls.white_chars}',
            '-c', f'tessedit_char_blacklist={cls.black_chars}',
        ]
        return config

    def __init__(self):
        super().__init__()
        self.tesseract_lang = 'm360'

    def preprocess(self, image_fp):
        image = super().preprocess(image_fp)

        # Remove random color dots, keep only black text, change background color to white.
        image = image.convert('RGB')
        for y in range(image.height):
            for x in range(image.width):
                c = image.getpixel((x, y))
                if c != (0, 0, 0):
                    image.putpixel((x, y), (255, 255, 255))

        return image

    def image_request_args(self):
        return {
            'url': 'https://crm.360.cn/login/checkImage',
            'params': {
                'type': 'login',
                'rnd': int(time.mktime(datetime.now().timetuple())),
            },
        }

    def validate_captcha_input(self, captcha):
        if len(captcha) != 5:
            raise ValueError('number of characters is not 5')

        invalid_chars = set(captcha) - self.charset
        if invalid_chars:
            raise ValueError(f'contains invalid chars {invalid_chars}')
