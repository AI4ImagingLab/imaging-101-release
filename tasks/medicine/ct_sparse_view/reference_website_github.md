# Reference Materials

## Core Algorithms

- **Chambolle-Pock (PDHG):** Chambolle, A. and Pock, T. (2011). "A first-order primal-dual algorithm for convex problems with applications to imaging." *JMIV*, 40(1), 120-145. https://doi.org/10.1007/s10851-010-0251-1
- **TV-regularized CT:** Sidky, E.Y. and Pan, X. (2008). "Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization." *PMB*, 53(17), 4777. https://doi.org/10.1088/0031-9155/53/17/021

## Software

- **scikit-image radon/iradon:** https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html
- **DIVal (Deep Inversion Validation Library):** https://github.com/jleuschn/dival — provides CT benchmark datasets and reconstruction baselines (our task uses the same Shepp-Logan phantom paradigm but implements transforms via scikit-image for CPU portability)
- **ASTRA Toolbox:** https://www.astra-toolbox.com/ — GPU-accelerated CT reconstruction (not used here due to macOS/CPU constraint, but the standard reference for production CT)

## Phantom

- **Shepp-Logan phantom:** Shepp, L.A. and Logan, B.F. (1974). "The Fourier reconstruction of a head section." *IEEE TNS*, 21(3), 21-43.
