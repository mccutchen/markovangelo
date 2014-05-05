#!/usr/bin/env python

# Example usage:
#
# ./markovangelo.py -o output --show --width=800 --height=300 --ngram-size=8 input/monalisa.gif  # noqa

import argparse
import collections
import itertools
import logging
import os
import sys
import random
import time

from PIL import Image
import vokram


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
    fill(out_w, out_h, target_pix, pix_stream)
    return img.crop((1, 1, out_w - 1, out_h - 1))


def prep_image(path):
    return Image.open(path).quantize(colors=256).convert('RGB')


def fill(w, h, target_pix, pix_stream):
    patchwork_fill(w, h, target_pix, pix_stream)


def simple_fill(w, h, target_pix, pix_stream):
    for y in range(0, h):
        for x in range(0, w):
            target_pix[x, y] = next(pix_stream)


def less_simple_fill(w, h, target_pix, pix_stream):
    for y in range(2, h - 2, 2):
        for x in range(2, w - 2, 2):
            target_pix[x, y] = next(pix_stream)
            for nx, ny in neighbors(x, y):
                target_pix[nx, ny] = next(pix_stream)


def stride_fill(w, h, target_pix, pix_stream):
    stride = int(h * .1)
    for row_y in range(0, h, stride):
        for x in range(0, w):
            for y in range(row_y, min(row_y + stride, h)):
                target_pix[x, y] = next(pix_stream)


def random_stride_fill(w, h, target_pix, pix_stream):
    base_stride = int(h * .05)
    stride_range = int(base_stride * .33)
    row_y = 0
    while row_y < h:
        stride = base_stride + random.randint(-stride_range, stride_range)
        for x in range(0, w):
            for y in range(row_y, min(row_y + stride, h)):
                target_pix[x, y] = next(pix_stream)
        row_y += stride


def flood_fill(w, h, target_pix, pix_stream):
    visited = set()
    q = collections.deque()
    q.append((0, 0))

    while q:
        x, y = q.popleft()
        if not 0 <= x < w or not 0 <= y < h or (x, y) in visited:
            continue

        visited.add((x, y))
        target_pix[x, y] = next(pix_stream)

        coords = [
            (x + 1, y),
            (x, y + 1)
        ]
        q.extend(coords)


def patchwork_fill(w, h, target_pix, pix_stream):
    # This fill breaks the images into square patches and fills each patch
    # individually.
    patch_size = int(max(w, h) * 0.025)
    patch_x_range = xrange(0, w, patch_size)
    patch_y_range = xrange(0, h, patch_size)
    patch_coords = list(itertools.product(patch_y_range, patch_x_range))
    patch_coords = sorted(patch_coords, reverse=True)

    for patch_y, patch_x in patch_coords:
        x_range = xrange(patch_x, min(patch_x + patch_size, w))
        y_range = xrange(patch_y, min(patch_y + patch_size, h))
        patch_pixels = list(itertools.product(x_range, y_range))
        patch_pixels = sorted(patch_pixels, key=lambda (x, y): (y * x))
        for x, y in patch_pixels:
            target_pix[x, y] = next(pix_stream)


def tokenize(w, h, pix):
    """We tokenize an image such that there is a token for each pixel
    and each of its neighboring pixels, so that each neighbor is
    equally likely to occur after any given pixel.

    (And we ignore the outermost pixels for simplicity's sake.)
    """
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for nx, ny in neighbors(x, y):
                yield pix[x, y]
                yield pix[nx, ny]


def neighbors(x, y):
    return [
        (x - 1, y),
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
    ]


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
