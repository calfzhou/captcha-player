#!/usr/bin/env python3
import importlib
import os
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
        try:
            self.validate_func(value)
            return value
        except Exception as e:
            value = click.prompt(f'Invalid captcha ({e}), input again', default=self.default)
            return self.convert(value, param, ctx)


@click.group(context_settings={'show_default': True})
@click.option('--captcha', 'captcha_name', default='base', help='which captcha class to play with')
@click.option('--lang', help='use a non-default tesseract language, e.g. eng, eng_best, eng_fast')
@click.option('--label-root', default='data/labeling', help='labeling data root folder path')
@click.pass_context
def main(ctx, captcha_name, lang, label_root):
    """Play with CAPTCHA."""
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['CAPTCHA_NAME'] = captcha_name
    ctx.obj['LABEL_ROOT'] = os.path.join(label_root, captcha_name)

    module = importlib.import_module(f'captcha.{captcha_name}')
    captcha_class_name = f'{inflection.camelize(captcha_name)}Captcha'
    recognizer = getattr(module, captcha_class_name)()
    if lang is not None:
        recognizer.tesseract_lang = lang

    ctx.obj['RECOGNIZER'] = recognizer


@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.option('--preview/--no-preview', default=False, help='whether preview the image')
@click.pass_context
def test(ctx, image, preview):
    """Try recognize given image."""
    recognizer = ctx.obj['RECOGNIZER']
    recognizer.preview_enabled = preview
    text = recognizer.recognize(image)
    print(text)


@click.command()
@click.option('-n', '--total', default=10,
              help='number of new images to fetch and label (0 for unlimited)')
@click.option('--overwrite/--no-overwrite', default=False,
              help='whether overwrite existing image for the same captcha')
@click.option('--preview/--no-preview', default=True, help='whether show image automatically')
@click.pass_context
def label(ctx, total, overwrite, preview):
    """Crawl and label training data."""
    label_root = ctx.obj['LABEL_ROOT']
    recognizer = ctx.obj['RECOGNIZER']
    recognizer.preview_enabled = preview
    os.makedirs(label_root, exist_ok=True)

    count = 0
    while total == 0 or count < total:
        count += 1
        request_args = recognizer.image_request_args()
        response = requests.get(**request_args)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_filename = f.name
            f.write(response.content)

        captcha = recognizer.recognize(temp_filename)
        captcha_type = CaptchaPromptType(recognizer.validate_captcha_input, default=captcha)
        captcha = click.prompt(
            f'[{count} / {total}] Enter captcha of image {temp_filename}', type=captcha_type, default=captcha)
        image_filename = os.path.join(label_root, f'{captcha}.png')
        if not overwrite and os.path.exists(image_filename):
            image_filename = os.path.join(label_root, f'{captcha}.{uuid.uuid4().hex[:4]}.png')

        print('move labeled image to', image_filename)
        os.replace(temp_filename, image_filename)


@click.command()
@click.option('--train-root', default='data/training', help='training data root folder path')
@click.pass_context
def truth(ctx, train_root):
    """Build ground truth data for tesseract training.

    It generates cleaned images and labeled transcripts.

    To list all possible characters appear in captcha, run:
    $ cat $TRAIN_ROOT/*.gt.txt | grep -o . | sort | uniq
    """
    captcha_name = ctx.obj['CAPTCHA_NAME']
    recognizer = ctx.obj['RECOGNIZER']
    label_root = ctx.obj['LABEL_ROOT']
    train_root = os.path.join(train_root, f'{captcha_name}-ground-truth')
    os.makedirs(train_root, exist_ok=True)

    for folder, _, filenames in os.walk(label_root):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext != recognizer.image_ext:
                continue

            captcha = basename.split(recognizer.label_filename_sep, 1)[0]
            label_image_path = os.path.join(folder, filename)
            train_basename = os.path.splitext(label_image_path)[0]
            train_basename = os.path.relpath(train_basename, folder).replace(os.path.sep, '-')
            train_image_path = os.path.join(train_root, f'{train_basename}.png')
            transcript_path = os.path.join(train_root, f'{train_basename}.gt.txt')

            print('labeled:', label_image_path, f'[{captcha}]')
            print('      =>', train_image_path)
            print('      =>', transcript_path)
            image = recognizer.preprocess(label_image_path)
            image.save(train_image_path)
            with open(transcript_path, 'w') as f:
                f.write(captcha)
                f.write('\n')

    train_hint = dedent(f"""
        To train, copy {train_root} folder to $TESSTRAIN_HOME/data, then run:

        $ cd $TESSTRAIN_HOME
        $ make clean MODEL_NAME={captcha_name}
        $ make training MODEL_NAME={captcha_name} PSM=7 START_MODEL={recognizer.train_start_model} TESSDATA="{TESSDATA_PATH}"
        $ cp "$TESSTRAIN_HOME/data/{captcha_name}.trainneddata" "{TESSDATA_PATH}/"
    """)
    print(train_hint)


@click.command()
@click.pass_context
def evaluate(ctx):
    """Evaluate trained model through all labeling images."""
    label_root = ctx.obj['LABEL_ROOT']
    recognizer = ctx.obj['RECOGNIZER']

    total = 0
    correct = 0
    for folder, _, filenames in os.walk(label_root):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext != '.png':
                continue

            captcha = basename.split('_', 1)[0]
            label_image_path = os.path.join(folder, filename)
            result = recognizer.recognize(label_image_path)

            total += 1
            if result == captcha:
                correct += 1
            else:
                print(f'expected: {captcha}, actual: {result}, image: {label_image_path}')

    precision = correct / total * 100 if total > 0 else 0
    print(f'evaluation result: {correct} of {total} [{precision:.1f}%] are correct')


main.add_command(test)
main.add_command(label)
main.add_command(truth)
main.add_command(evaluate)

if __name__ == '__main__':
    main()
