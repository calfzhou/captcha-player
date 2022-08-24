#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import tempfile
from textwrap import dedent
import uuid

import click
import inflection
import requests

from captcha.base import TESSDATA_PATH


class CaptchaPromptType(click.ParamType):
    name = 'captcha'

    def __init__(self, validate_func, default=None):
        self.validate_func = validate_func
        self.default = default

    def convert(self, value, param, ctx):
        if value == 'skip':
            return None

        try:
            self.validate_func(value)
            return value
        except Exception as e:
            value = click.prompt(
                f'Invalid captcha ({e}), input again (enter `skip` to skip)',
                default=self.default)
            return self.convert(value, param, ctx)


@click.group(context_settings={'show_default': True})
@click.option('--class', '_class', default='base', help='which captcha class to play with')
@click.option('--lang', help='use a non-default tesseract language, e.g. eng, eng_best, eng_fast')
@click.option('--label-root', type=click.Path(file_okay=False, writable=True, path_type=Path),
              default='data/labeling', help='labeling data root folder path')
@click.pass_context
def main(ctx, _class: str, lang: str, label_root: Path):
    """Play with CAPTCHA."""
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['CLASS_NAME'] = _class
    ctx.obj['LABEL_ROOT'] = label_root.joinpath(_class)

    module = importlib.import_module(f'captcha.{_class}')
    captcha_class_name = f'{inflection.camelize(_class)}Captcha'
    recognizer = getattr(module, captcha_class_name)()
    if lang is not None:
        recognizer.tesseract_lang = lang

    ctx.obj['RECOGNIZER'] = recognizer


@main.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--preview/--no-preview', default=False, help='whether preview the image')
@click.pass_context
def test(ctx, image: Path, preview: bool):
    """Try recognize given image."""
    recognizer = ctx.obj['RECOGNIZER']
    recognizer.preview_enabled = preview
    text = recognizer.recognize(image)
    print(text)


@main.command()
@click.option('-n', '--total', default=10,
              help='number of new images to fetch and label (0 for unlimited)')
@click.option('--overwrite/--no-overwrite', default=False,
              help='whether overwrite existing image for the same captcha')
@click.option('--preview/--no-preview', default=True, help='whether show image automatically')
@click.pass_context
def label(ctx, total: int, overwrite: bool, preview: bool):
    """Crawl and label training data."""
    recognizer = ctx.obj['RECOGNIZER']
    recognizer.preview_enabled = preview
    label_root: Path = ctx.obj['LABEL_ROOT']
    label_root.mkdir(parents=True, exist_ok=True)

    count = 0
    while total == 0 or count < total:
        count += 1
        request_args = recognizer.image_request_args()
        response = requests.get(**request_args)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_filepath = Path(f.name)
            f.write(response.content)

        captcha = recognizer.recognize(temp_filepath)
        captcha_type = CaptchaPromptType(recognizer.validate_captcha_input, default=captcha)
        captcha = click.prompt(
            f'[{count} / {total}] Enter captcha of {temp_filepath} (enter `skip` to skip)',
            type=captcha_type, default=captcha)
        if captcha is None:
            print('skipped')
            temp_filepath.unlink()
            count -= 1
            continue

        image_path = label_root.joinpath(captcha).with_suffix(recognizer.image_ext)
        if not overwrite and image_path.exists():
            new_stem = f'{captcha}{recognizer.label_filename_sep}{uuid.uuid4().hex[:4]}'
            image_path = image_path.with_stem(new_stem)

        print('move labeled image to', image_path)
        temp_filepath.replace(image_path)


@main.command()
@click.option('--train-root', type=click.Path(file_okay=False, writable=True, path_type=Path),
              default='data/training', help='training data root folder path')
@click.pass_context
def truth(ctx, train_root: Path):
    """Build ground truth data for tesseract training.

    It generates cleaned images and labeled transcripts.

    To list all possible characters appear in captcha, run:
    $ cat $TRAIN_ROOT/*.gt.txt | grep -o . | sort | uniq
    """
    class_name: str = ctx.obj['CLASS_NAME']
    recognizer = ctx.obj['RECOGNIZER']
    label_root: Path = ctx.obj['LABEL_ROOT']
    train_root = train_root.joinpath(f'{class_name}-ground-truth')
    train_root.mkdir(parents=True, exist_ok=True)

    for folder, _, filenames in os.walk(label_root):
        folder = Path(folder)
        for filename in filenames:
            basename, ext = (p := Path(filename)).stem, p.suffix
            if ext != recognizer.image_ext:
                continue

            captcha = basename.split(recognizer.label_filename_sep, 1)[0]
            label_image_path = folder.joinpath(filename)
            formatted_name = str(label_image_path.relative_to(label_root)).replace(os.path.sep, '-')
            train_image_path = train_root.joinpath(formatted_name).with_suffix('.png')
            transcript_path = train_image_path.with_suffix('.gt.txt')

            print(f'labeled:', label_image_path, f'[{captcha}]')
            print('      =>', train_image_path)
            print('      =>', transcript_path)
            image = recognizer.preprocess(label_image_path)
            image.save(train_image_path)
            with transcript_path.open('w') as f:
                f.write(captcha)
                f.write('\n')

    train_hint = dedent(f"""
        To train, copy {train_root} folder to $TESSTRAIN_HOME/data, then run:

        $ cd $TESSTRAIN_HOME
        $ make clean MODEL_NAME={class_name}
        $ make training MODEL_NAME={class_name} PSM=7 START_MODEL={recognizer.train_start_model} TESSDATA="{TESSDATA_PATH}"
        $ cp "$TESSTRAIN_HOME/data/{class_name}.trainneddata" "{TESSDATA_PATH}/"
    """)
    print(train_hint)


@main.command()
@click.pass_context
def evaluate(ctx):
    """Evaluate trained model through all labeling images."""
    label_root: Path = ctx.obj['LABEL_ROOT']
    recognizer = ctx.obj['RECOGNIZER']

    total = 0
    correct = 0
    for folder, _, filenames in os.walk(label_root):
        folder = Path(folder)
        for filename in filenames:
            basename, ext = (p := Path(filename)).stem, p.suffix
            if ext != recognizer.image_ext:
                continue

            captcha = basename.split(recognizer.label_filename_sep, 1)[0]
            label_image_path = folder.joinpath(filename)
            result = recognizer.recognize(label_image_path)

            total += 1
            if recognizer.case_sensitive and result == captcha:
                correct += 1
            elif not recognizer.case_sensitive and result.upper() == captcha.upper():
                correct += 1
            else:
                print(f'expected: {captcha}, actual: {result}, image: {label_image_path}')

    precision = correct / total * 100 if total > 0 else 0
    print(f'evaluation result: {correct} of {total} [{precision:.1f}%] are correct')


if __name__ == '__main__':
    main()
