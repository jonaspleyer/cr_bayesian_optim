import numpy as np
import skimage as sk

if __name__ == "__main__":
    # Load Image
    img = sk.io.imread("paper/figures/cells_at_iter_0000060200.png")

    # Reduce use most common color
    img_dim_new = np.zeros((2, 2, 4))

    nx = img.shape[0] / img_dim_new.shape[0]
    ny = img.shape[1] / img_dim_new.shape[1]
    for i in range(img_dim_new.shape[0]):
        for j in range(img_dim_new.shape[1]):
            nx_low = int(np.round(i * nx))
            nx_high = int(np.round((i + 1) * nx) + 1)

            ny_low = int(np.round(j * ny))
            ny_high = int(np.round((j + 1) * ny) + 1)

            idents, counts = np.unique(
                img[nx_low:nx_high, ny_low:ny_high].reshape((-1, 4)),
                axis=0,
                return_counts=True,
            )

            print(idents[:3], len(counts), np.sum(counts))
