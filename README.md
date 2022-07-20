# captcha-player

Play with captchas: labeling, training, and evaluating.

## Prerequisites

Install [tesseract](https://github.com/tesseract-ocr/tesseract) with [training tools](https://github.com/tesseract-ocr/tesstrain).

Assume that [tesstrain](https://github.com/tesseract-ocr/tesstrain) is located in `$TESSTRAIN_HOME`.

## Installation

``` bash
pip install -r requirements.txt -c constraints.txt
```

## Play

General usage:

``` text
Usage: play_captcha.py [OPTIONS] COMMAND [ARGS]...

  Play with CAPTCHA.

Options:
  --captcha TEXT     which captcha class to play with  [default: base]
  --lang TEXT        use a non-default tesseract language, e.g. eng, eng_best,
                     eng_fast
  --label-root TEXT  labeling data root folder path  [default: data/labeling]
  --help             Show this message and exit.

Commands:
  evaluate  Evaluate trained model through all labeling images.
  label     Crawl and label training data.
  test      Try recognize given image.
  truth     Build ground truth data for tesseract training.
```

### Test / Recognize Single Image

``` text
Usage: play_captcha.py test [OPTIONS] IMAGE

  Try recognize given image.

Options:
  --preview / --no-preview  whether preview the image  [default: no-preview]
  --help                    Show this message and exit.
```

### Labeling

``` text
Usage: play_captcha.py label [OPTIONS]

  Crawl and label training data.

Options:
  -n, --total INTEGER           number of new images to fetch and label (0 for
                                unlimited)  [default: 10]
  --overwrite / --no-overwrite  whether overwrite existing image for the same
                                captcha  [default: no-overwrite]
  --preview / --no-preview      whether show image automatically  [default:
                                preview]
  --help                        Show this message and exit.
```

### Prepare Training Data

``` text
Usage: play_captcha.py truth [OPTIONS]

  Build ground truth data for tesseract training.

  It generates cleaned images and labeled transcripts.

  To list all possible characters appear in captcha, run: $ cat
  $TRAIN_ROOT/*.gt.txt | grep -o . | sort | uniq

Options:
  --train-root TEXT  training data root folder path  [default: data/training]
  --help             Show this message and exit.
```

When ground truth data is generated, check command output for how to train model.

### Evaluating (with all labelled images)

``` text
Usage: play_captcha.py evaluate [OPTIONS]

  Evaluate trained model through all labeling images.

Options:
  --help  Show this message and exit.
```
