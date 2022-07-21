
import os

from PIL import Image
import pytesseract

TESSDATA_PATH = os.path.join(os.path.dirname(__file__), 'tessdata')


class BaseCaptcha:
    @classmethod
    @property
    def tesseract_config_array(cls):
        config = [
            '--tessdata-dir', f'"{TESSDATA_PATH}"',
            '--psm', '7', # Treat the image as a single text line.
        ]
        return config

    @classmethod
    def _build_tesseract_config_string(cls):
        return ' '.join(cls.tesseract_config_array)

    def __init__(self):
        self.preview_enabled = False
        self.preview_scale = 1
        self.tesseract_lang = 'eng'
        self.tesseract_config = self._build_tesseract_config_string()
        self.train_start_model = 'eng_best'
        self.label_filename_sep = '_'
        self.image_ext = '.png'

    def preprocess(self, image_fp):
        image = Image.open(image_fp)
        if self.preview_enabled:
            self._preview(image)

        return image

    def recognize(self, image_fp):
        image = self.preprocess(image_fp)
        text = pytesseract.image_to_string(image, lang=self.tesseract_lang, config=self.tesseract_config)
        text = text.strip() # There might be an extra `\n` at the end of the string.
        return text

    def image_request_args(self):
        raise NotImplementedError

    def validate_captcha_input(self, captcha):
        # Implement custom validation in subclasses.
        pass

    def _preview(self, image):
        if self.preview_scale != 1:
            image = image.resize((image.width * self.preview_scale, image.height * self.preview_scale))

        image.show()

