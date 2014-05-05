#!/usr/bin/env python

# Example usage:
#
# ./markovangelo.py -o output --show --width=800 --height=300 --ngram-size=8 input/monalisa.gif  # noqa

import argparse
import itertools
import logging
import os
import sys
import time

from PIL import Image, ImageDraw
import vokram

import fills
import utils


def remix(paths, ngram_size, output_size):
    imgs = map(prep_image, paths)
    out_w, out_h = output_size

    tokens_iters = []
    for img in imgs:
        w, h = img.size
        tokens_iters.append(tokenize(w, h, img.load()))

    sentinal = 0
    tokens = itertools.chain.from_iterable(tokens_iters)
    model = vokram.build_model(tokens, ngram_size, sentinal)
    # start_key = (sentinal,) * ngram_size
    start_key = None

    img_count = len(imgs)
    pixels = sum(img.size[0] * img.size[1] for img in imgs)
    logging.info('%d image(s), %d pixels', img_count, pixels)
    logging.info('Model size: %d', len(model))

    img = Image.new('RGB', output_size)
    target_pix = img.load()

    pix_stream = vokram.markov_chain(model, start_key=start_key)
    fill(out_w, out_h, target_pix, pix_stream, ImageDraw.Draw(img))
    return img.crop((1, 1, out_w - 1, out_h - 1))


def prep_image(path):
    return Image.open(path).quantize(colors=256).convert('RGB')


def fill(w, h, target_pix, pix_stream, draw):
    fills.patchwork_fill(w, h, target_pix, pix_stream, draw)


def tokenize(w, h, pix):
    """We tokenize an image such that there is a token for each pixel
    and each of its neighboring pixels, so that each neighbor is
    equally likely to occur after any given pixel.

    (And we ignore the outermost pixels for simplicity's sake.)
    """
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for nx, ny in utils.neighbors(x, y):
                yield pix[x, y]
                yield pix[nx, ny]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        prog='markovangelo',
        description='Uses Markov chains to remix images.')
    arg_parser.add_argument(
        '-n', '--ngram-size', type=int, default=4)
    arg_parser.add_argument(
        '--width', type=int, required=True,
        help='Output image width')
    arg_parser.add_argument(
        '--height', type=int, required=True,
        help='Output image height')
    arg_parser.add_argument(
        '-o', '--output-dir',
        help='Optional output dir. If given, a path will be chosen for you.')
    arg_parser.add_argument(
        '--show', action='store_true', help='Open result in image viewer')
    arg_parser.add_argument(
        'source_file', nargs='+', help='Input image(s)')

    args = arg_parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    img = remix(args.source_file, args.ngram_size, (args.width, args.height))
    if args.show:
        img.show()
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        filename = '%d.png' % time.time()
        outpath = os.path.join(args.output_dir, filename)
        logging.info(os.path.abspath(outpath))
        outfile = open(outpath, 'wb')
    else:
        outfile = sys.stdout
    img.save(outfile, 'png')
