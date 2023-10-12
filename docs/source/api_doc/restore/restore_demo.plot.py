from imgutils.data import load_image
from imgutils.restore.nafnet import restore_with_nafnet
from imgutils.restore.scunet import restore_with_scunet
from plot import image_plot

if __name__ == '__main__':
    img_q35 = load_image('sample/jpg-q35.jpg')
    img_gnoise = load_image('sample/gnoise.png')

    image_plot(
        (img_q35, 'JPEG Quality35'),
        (restore_with_nafnet(img_q35), 'JPEG Quality35\n(Fixed By NafNet)'),
        (img_gnoise, 'Gaussian Noise'),
        (restore_with_scunet(img_gnoise), 'Gaussian Noise\n(Fixed By SCUNet)'),
        columns=2,
        figsize=(6, 8),
    )
