from imgutils.data import load_image
from imgutils.segment import segment_rgba_with_isnetis
from plot import image_plot

if __name__ == '__main__':
    image_plot(
        'hutao.png',
        (segment_rgba_with_isnetis(load_image('hutao.png'))[1], 'hutao_seg.png'),
        'skadi.jpg',
        (segment_rgba_with_isnetis(load_image('skadi.jpg'))[1], 'skadi_seg.png'),
        columns=2,
        figsize=(8, 9),
    )
