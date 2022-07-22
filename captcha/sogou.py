
from datetime import datetime
import string
import time

from .base import BaseCaptcha


class SogouCaptcha(BaseCaptcha):
    white_chars = f'{string.digits}{string.ascii_uppercase}{string.ascii_lowercase}'
    black_chars = '01Ilo'
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
        self.tesseract_lang = 'sogou'
        self.image_ext = '.jpeg'

    def preprocess(self, image_fp):
        image = super().preprocess(image_fp)

        # Convert to grey level image, remove light shadows.
        image = image.convert('L')
        threshold = 64 # Pixels lighter than threshold will be converted to white.
        table = list(range(threshold)) + [255] * (256 - threshold)
        image = image.point(table)

        return image

    def recognize(self, image_fp):
        text = super().recognize(image_fp)
        extra = len(text) - 4
        if extra > 0:
            text = self._fix_diplopia(text, extra)

        return text

    def _fix_diplopia(self, text, max_tries):
        """https://github.com/tesseract-ocr/tesseract/issues/3477"""
        ambiguous = 'ckpsvwxyz'
        remains = []

        prev = None
        for ch in text:
            if max_tries > 0 and prev and prev.lower() in ambiguous and prev.lower() == ch.lower() and prev != ch:
                max_tries -= 1
                prev = None
            else:
                remains.append(ch)
                prev = ch

        return ''.join(remains)

    def image_request_args(self):
        t = datetime.now()
        nounce = int(time.mktime(t.timetuple()) * 1000) + t.microsecond // 1000
        return {
            'url': f'https://auth.p4p.sogou.com/validateCode/{nounce}',
            'params': {
                'code': 'checkcode',
                'nounce': nounce,
            },
        }

    def validate_captcha_input(self, captcha):
        if len(captcha) != 4:
            raise ValueError('incorrect number of characters')

        invalid_chars = set(captcha) - self.charset
        if invalid_chars:
            raise ValueError(f'contains invalid chars {invalid_chars}')
