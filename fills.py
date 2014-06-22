import collections
import itertools
import logging
import math
import random

import utils


def simple_fill(w, h, target_pix, pix_stream):
    for y in range(0, h):
        for x in range(0, w):
            target_pix[x, y] = next(pix_stream)


def less_simple_fill(w, h, target_pix, pix_stream):
    for y in range(2, h - 2, 2):
        for x in range(2, w - 2, 2):
            target_pix[x, y] = next(pix_stream)
            for nx, ny in utils.neighbors(x, y):
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


def patch_walk(w, h, patch_size):
    visited = set()
    heading = (1, 0)
    pos = (0, 0)

    def invalid_pos((x, y)):
        return (x, y) in visited or \
            x < 0 or x + patch_size > w or \
            y < 0 or y + patch_size > h

    def update_pos(pos, heading):
        x, y = pos
        dx, dy = heading
        next_pos = (x + (patch_size * dx), y + (patch_size * dy))
        return (x, y), next_pos

    turns = {
        (1, 0): (0, 1),
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
    }

    while True:
        yield pos
        visited.add(pos)
        last_pos, pos = update_pos(pos, heading)
        if invalid_pos(pos):
            logging.warn(' ! new direction: %r %r', pos, heading)
            heading = turns[heading]
            _, pos = update_pos(last_pos, heading)
            if invalid_pos(pos):
                logging.warn(' !! no valid path? %r %r', pos, heading)
                break


def patchwork_fill(w, h, target_pix, pix_stream, draw):
    # This fill breaks the images into square patches and fills each patch
    # individually.
    patch_size = int(max(w, h) * 0.025)

    cx = int(w * .66)
    cy = h / 2
    hypot = math.hypot
    hypot_sort = lambda (x, y): hypot(x - cx, y - cy)

    patch_coords = list(patch_walk(w, h, patch_size))

    for patch_x, patch_y in patch_coords:
        start = (patch_x, patch_y)
        end = (min(patch_x + patch_size, w),
               min(patch_y + patch_size, h))
        patch_pixels = precalculate_coords(start, end, sort=hypot_sort)
        for x, y in patch_pixels:
            target_pix[x, y] = next(pix_stream)

    last_coord = None
    p_off = patch_size / 2
    offset = lambda (x, y): (x + p_off, y + p_off)
    for i, coord in enumerate(patch_coords):
        if last_coord:
            draw.line(offset(last_coord) + offset(coord), fill='pink')
        last_coord = coord


def circular_fill(w, h, target_pix, pix_stream):
    # This fill precalculates the list of (x, y) coordinates in the target
    # image and sorts them based on their distance from the center before
    # filling based on that order.
    cx = w / 2
    cy = h / 2
    hypot = math.hypot
    sort = lambda (x, y): hypot(x - cx, y - cy)
    coords = precalculate_coords((w, h), sort=sort)
    for x, y in coords:
        target_pix[x, y] = next(pix_stream)


def radial_fill(w, h, target_pix, pix_stream):
    # This fill precalculates the list of (x, y) coordinates in the target
    # image and sorts them based on their distance from the center before
    # filling based on that order.
    cx = w / 2
    cy = h / 2
    atan2 = math.atan2
    sort = lambda (x, y): atan2(y - cy, x - cx)
    coords = precalculate_coords((w, h), sort=sort)
    for x, y in coords:
        target_pix[x, y] = next(pix_stream)


def voronoi_fill(w, h, target_pix, pix_stream):
    centroid_count = int(w * h * 0.00025)
    centroids = [(random.randint(0, w), random.randint(0, h))
                 for _ in xrange(centroid_count)]
    centroid_map = dict(enumerate(centroids))

    def sort((x, y), hypot=math.hypot):
        dists = ((key, hypot(x-cx, y-cy))
                 for key, (cx, cy) in centroid_map.iteritems())
        key, _ = min(dists, key=lambda x: x[1])
        return key

    all_coords = precalculate_coords((w, h), sort=sort)
    grouped_coords = itertools.groupby(all_coords, sort)
    for centroid_key, coords in grouped_coords:
        if centroid_key % 2 == 0:
            key = lambda (x, y): y - x
        else:
            key = lambda (x, y): x * y
        coords = sorted(coords, key=key)
        for x, y in coords:
            target_pix[x, y] = next(pix_stream)


def precalculate_coords(start, end=None, step=1, sort=None):
    if end:
        x0, y0 = start
        x1, y1 = end
    else:
        x0 = y0 = 0
        x1, y1 = start
    x_range = xrange(x0, x1, step)
    y_range = xrange(y0, y1, step)
    coords = itertools.product(x_range, y_range)
    return sorted(coords, key=sort) if callable(sort) else coords
