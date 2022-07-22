

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

        # TODO: Clean image.

        return image

    def validate_captcha_input(self, captcha):
        if len(captcha) != 5:
            raise ValueError('number of characters is not 5')

        invalid_chars = set(captcha) - self.charset
        if invalid_chars:
            raise ValueError(f'contains invalid chars {invalid_chars}')
