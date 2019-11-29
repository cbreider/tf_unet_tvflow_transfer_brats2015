import numpy as np
import matplotlib.pyplot as plt


arr_compare = np.zeros((8*240,8*240), dtype=np.float)
for i in range(8):
    for j in range(8):
        arr_compare[i*240:(i+1)*240, j*240:(j+1)*240] = np.ones((240,240), dtype=np.float) * (float(i)*8.0+float(j)) / (64.0)


arr = np.zeros((240,240, 64), dtype=np.float)
for i in range(64):
    arr[:, :, i] = np.ones((240,240), dtype=np.float) * (float(i)) / (64.0)

BSZ = [8,8] # Block size
p,q = BSZ
arr = arr.reshape(240,240,8,8).transpose(2,0,3,1).reshape(8*240,8*240)

plt.matshow(arr_compare)
plt.matshow(arr)
plt.show()


def tv_denoise(img, weight=0.1, tau=0.125, eps=1e-3, num_iter_max=200, ):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more de-noising (at
        the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)
    nm = np.prod(img.shape[:2])

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        a =np.sqrt(nm)
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1

    u = u - u.min()
    u = u / u.max()
    return u