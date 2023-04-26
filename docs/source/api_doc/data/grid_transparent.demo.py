from plot import image_plot

if __name__ == '__main__':
    image_plot(
        'nian.png',
        'dori.png',
        save_as='grid_transparent.dat.svg',
        columns=2,
        figsize=(12, 16),
    )
