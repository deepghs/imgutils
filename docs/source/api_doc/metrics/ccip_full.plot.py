from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *(f'ccip/{i}.jpg' for i in range(1, 13)),
        columns=3,
        figsize=(8, 12),
    )
