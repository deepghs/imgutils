from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *(f'lpips/{i}.jpg' for i in range(1, 10)),
        save_as='lpips_full.dat.svg',
        columns=3,
        figsize=(6, 8),
    )
