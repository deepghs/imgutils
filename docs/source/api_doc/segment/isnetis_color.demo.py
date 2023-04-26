from imgutils.data import load_image
from imgutils.segment import segment_with_isnetis
from plot import image_plot

if __name__ == '__main__':
    image_plot(
        'hutao.png',
        (segment_with_isnetis(load_image('hutao.png'))[1], 'hutao_seg.png'),
        'skadi.jpg',
        (segment_with_isnetis(load_image('skadi.jpg'), background='white')[1], 'skadi_seg.jpg'),
        save_as='isnetis_color.dat.svg',
        columns=2,
        figsize=(8, 9),
    )
