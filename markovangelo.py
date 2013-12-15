
import Image
import vokram


def main():
    img = Image.open('monalisa.gif')
    w, h = img.size
    pix = img.load()
    tokens = tokenize(w, h, pix)
    model = vokram.build_model(tokens, 10)

    new_pix = vokram.markov_chain(model, w * h)
    new_pix_iter = iter(new_pix)
    for y in range(h):
        for x in range(w):
            pix[x, y] = next(new_pix_iter)

    img.show()


def tokenize(w, h, pix):
    for y in range(h):
        for x in range(w):
            yield pix[x, y]


if __name__ == '__main__':
    main()
