import glob
import os.path

from plot import image_plot

if __name__ == '__main__':
    images = sorted(glob.glob('aest/*.jpg'), key=lambda x: float(os.path.splitext(x)[0].split('-')[-1]))
    items = []
    for image in images:
        body, ext = os.path.splitext(image)
        seg1, seg2 = body.split('-')
        display_name = f'{seg1}{ext}\n(score: {seg2})'
        items.append((image, display_name))

    image_plot(
        *items,
        save_as='aesthetic_full.dat.svg',
        columns=4,
        figsize=(11, 9),
    )
