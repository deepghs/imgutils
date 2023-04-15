from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *['skadi.jpg', 'hutao.jpg'],
        save_as='tagging_demo.dat.svg',
        columns=2,
        figsize=(4, 8),
    )
