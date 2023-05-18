from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *['skadi.jpg', 'hutao.jpg'],
        columns=2,
        figsize=(4, 8),
    )
