from hbutils.reflection import nested_for
from tqdm.auto import tqdm

from imgutils.edge import edge_image_with_canny, edge_image_with_lineart, edge_image_with_lineart_anime
from plot import image_table

if __name__ == '__main__':
    demo_images = ['hutao.png', 'skadi.jpg', 'xx.jpg', 'surtr_underwear.png']
    funcs = [
        ('canny', edge_image_with_canny),
        ('lineart', edge_image_with_lineart),
        ('lineart\nanime', edge_image_with_lineart_anime)
    ]
    table = [[item] for item in demo_images]
    for (xi, origin_image), (_, func) in \
            tqdm(list(nested_for(enumerate(demo_images), funcs))):
        table[xi].append(func(origin_image, backcolor='white', forecolor='black'))

    image_table(
        table,
        columns=['origin', *(name for name, _ in funcs)],
        rows=['' for _ in demo_images],
        save_as='edge.dat.svg',
        figsize=(1800, 1350),
    )
