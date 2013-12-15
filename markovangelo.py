import collections
import Image
import vokram


def main():
    img = Image.open('monalisa.gif')
    w, h = img.size
    pix = img.load()
    tokens = tokenize(w, h, pix)
    model = vokram.build_model(tokens, 10)

    new_pix = vokram.markov_chain(model, w * h)
    flood_fill(w / 2, h / 2, w, h, pix, iter(new_pix))
    img.show()


def flood_fill(w, h, pix, new_pix):
    visited = set()
    q = collections.deque()
    q.append((0, 0))

    while q:
        x, y = q.popleft()
        if (x, y) in visited or not 0 <= x < w or not 0 <= y < h:
            continue

        visited.add((x, y))
        pix[x, y] = next(new_pix)

        q.append((x + 1, y))
        q.append((x + 1, y + 1))
        q.append((x, y + 1))


def tokenize(w, h, pix):
    for y in range(h):
        for x in range(w):
            yield pix[x, y]
    for x in range(w):
        for y in range(h):
            yield pix[x, y]


if __name__ == '__main__':
    main()
