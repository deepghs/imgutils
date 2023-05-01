from functools import partial

from hbutils.reflection import nested_for
from tqdm.auto import tqdm

from imgutils.edge import edge_image_with_lineart as edge_image
from plot import image_table

if __name__ == '__main__':
    demo_images = ['hutao.png', 'skadi.jpg']
    funcs = [
        ('lineart\ncoarse', partial(edge_image, coarse=True)),
        ('lineart\ncoarse\nbackcolor=transparent', partial(edge_image, backcolor='transparent', coarse=True)),
        ('lineart\ncoarse\nbackcolor=white\nforecolor=black',
         partial(edge_image, backcolor='white', forecolor='black', coarse=True)),
    ]
    table = [[item] for item in demo_images]
    for (xi, origin_image), (_, func) in \
            tqdm(list(nested_for(enumerate(demo_images), funcs))):
        table[xi].append(func(origin_image))

    image_table(
        table,
        columns=['origin', *(name for name, _ in funcs)],
        rows=['' for _ in demo_images],
        save_as='lineart_coarse.dat.svg',
        figsize=(1600, 980),
    )
