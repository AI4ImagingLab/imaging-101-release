import os
import numpy as np
import math
import time
import importlib

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


class MSSNDenoiser:
    """Multiple Self-Similarity Network denoiser.

    Pre-trained RNN denoiser using multi-head attention for exploiting
    non-local self-similarity. Operates on overlapping patches.
    """

    def __init__(self, image_shape, sigma=5,
                 model_checkpoints="models/checkpoints/mssn-550000iters",
                 patch_size=42, stride=7, state_num=8,
                 num_heads=2, key_dim=128, value_dim=128, batch_size=1):
        """Initialize MSSN denoiser and load pre-trained weights.

        Args:
            image_shape: tuple (N, M) — image dimensions
            sigma: int — noise level the denoiser was trained on
            model_checkpoints: str — path to TF checkpoint
            patch_size: int — size of square patches
            stride: int — stride between patches
            state_num: int — number of recurrent states
            num_heads: int — number of attention heads
            key_dim: int — dimension of attention keys
            value_dim: int — dimension of attention values
            batch_size: int — batch size for inference
        """
        self.nx, self.ny = image_shape[0], image_shape[1]
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

        tf.reset_default_graph()

        self.input_image = tf.placeholder(tf.float32, shape=(None, None, None))
        input_shape = tf.shape(self.input_image)
        input_reshaped = tf.reshape(
            self.input_image,
            [input_shape[0], input_shape[1], input_shape[2], 1],
        )

        self.output_image = _build_mssn(
            input_reshaped, state_num, batch_size,
            key_dim, value_dim, num_heads, is_training=False,
        )

        init_local = tf.local_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init_local)

        if tf.io.gfile.exists(model_checkpoints) or tf.io.gfile.exists(
            model_checkpoints + ".index"
        ):
            saver.restore(self.sess, model_checkpoints)
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_checkpoints}"
            )

    def denoise(self, image):
        """Denoise a single grayscale image using patch-based MSSN inference.

        The image is split into overlapping patches, each denoised independently,
        then overlapping outputs are averaged. Uses residual learning: output = input + residual.

        Args:
            image: ndarray (N, M) — input image in [0, 1]

        Returns:
            ndarray (N, M) — denoised image in [0, 1]
        """
        nx, ny = image.shape[0], image.shape[1]

        # Scale to [0, 255]
        noisy_img = (image * 255.0).astype(np.float32)

        # Extract overlapping patches
        h_idx_list = list(
            range(0, noisy_img.shape[0] - self.patch_size, self.stride)
        ) + [noisy_img.shape[0] - self.patch_size]
        w_idx_list = list(
            range(0, noisy_img.shape[1] - self.patch_size, self.stride)
        ) + [noisy_img.shape[1] - self.patch_size]

        patch_list = []
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                patch_list.append(
                    noisy_img[h_idx : h_idx + self.patch_size,
                              w_idx : w_idx + self.patch_size]
                )
        patches = np.stack(patch_list, axis=0)

        # Run inference in batches
        n_patches = patches.shape[0]
        batch_no = int(math.ceil(n_patches / float(self.batch_size)))
        output_patches_list = []

        for batch_id in range(batch_no):
            start = batch_id * self.batch_size
            end = min(start + self.batch_size, n_patches)
            cur_batch = patches[start:end]
            output_batch = self.sess.run(
                self.output_image, feed_dict={self.input_image: cur_batch}
            )
            output_patches_list.append(output_batch)

        output_patches = np.concatenate(output_patches_list, axis=0)

        # Average overlapping patches
        cnt_map = np.zeros_like(noisy_img)
        output_img = np.zeros_like(noisy_img)
        cnt = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                output_img[
                    h_idx : h_idx + self.patch_size,
                    w_idx : w_idx + self.patch_size,
                ] += output_patches[cnt, :, :, :].squeeze()
                cnt_map[
                    h_idx : h_idx + self.patch_size,
                    w_idx : w_idx + self.patch_size,
                ] += 1
                cnt += 1
        output_img /= cnt_map

        # Residual learning: denoised = input + network_output
        denoised = output_img.squeeze() + noisy_img

        return np.reshape(denoised, [nx, ny]) / 255.0


def pnp_pgm(forward_model, denoiser, y, num_iter=200, step=1.0,
             xtrue=None, verbose=True, save_dir=None):
    """Plug-and-Play Proximal Gradient Method for MRI reconstruction.

    Alternates gradient descent on the data fidelity term with
    denoising as the proximal step.

    Args:
        forward_model: MRIForwardModel — provides grad() method
        denoiser: MSSNDenoiser — provides denoise() method
        y: ndarray — k-space measurements
        num_iter: int — number of iterations
        step: float — gradient step size
        xtrue: ndarray or None — ground truth for SNR tracking
        verbose: bool — print iteration info
        save_dir: str or None — directory to save per-iteration results

    Returns:
        tuple: (reconstruction, history)
            reconstruction: ndarray (N, M) — final reconstructed image
            history: dict with keys 'snr', 'dist', 'time', 'relative_change'
    """
    compute_snr = lambda gt, est: 20 * np.log10(
        np.linalg.norm(gt.flatten("F"))
        / np.linalg.norm(gt.flatten("F") - est.flatten("F"))
    )
    compute_tol = lambda x, xnext: np.linalg.norm(
        x.flatten("F") - xnext.flatten("F")
    ) / np.linalg.norm(x.flatten("F"))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Initialize
    x = np.zeros(forward_model.sig_size, dtype=np.float32)
    p_dummy = np.zeros(forward_model.sig_size)

    snr_history = []
    dist_history = []
    time_history = []
    rel_change_history = []

    for k in range(num_iter):
        t_start = time.time()

        # Gradient step
        g, _ = forward_model.grad(x, y)
        s = np.clip(x - step * g, 0, np.inf)

        # Denoising proximal step
        xnext = denoiser.denoise(s)

        # Track convergence
        px = xnext
        dist_history.append(np.linalg.norm(x.flatten("F") - px.flatten("F")) ** 2)
        time_history.append(time.time() - t_start)

        if k == 0:
            rel_change_history.append(np.inf)
        else:
            rel_change_history.append(compute_tol(x, xnext))

        if xtrue is not None:
            snr_history.append(compute_snr(xtrue, x))

        x = xnext

        # Save per-iteration results
        if save_dir:
            import scipy.io as spio
            spio.savemat(
                os.path.join(save_dir, f"iter_{k + 1}_mat.mat"), {"img": xnext}
            )

        if verbose:
            snr_str = f"[snr: {snr_history[-1]:.2f}]" if xtrue is not None else ""
            print(
                f"[PnP-PGM: {k + 1}/{num_iter}]"
                f"[tol: {rel_change_history[-1]:.5e}]"
                f"[||x-Px||^2: {dist_history[-1]:.5e}]"
                f"[step: {step:.1e}]"
                f"[time: {np.sum(time_history):.1f}]"
                f"{snr_str}"
            )

    history = {
        "snr": snr_history,
        "dist": dist_history,
        "time": time_history,
        "relative_change": rel_change_history,
    }
    return x, history


# ---- MSSN model architecture (adapted from original models/mssn.py) ----

def _build_mssn(model_input, state_num, batch_size,
                key_dim, value_dim, num_heads, is_training=False):
    """Build the MSSN computation graph.

    Args:
        model_input: tf tensor (B, H, W, 1) — input patches
        state_num: int — number of recurrent states
        batch_size: int
        key_dim: int — attention key dimension
        value_dim: int — attention value dimension
        num_heads: int — number of attention heads
        is_training: bool

    Returns:
        tf tensor (B, H, W, 1) — denoised residual
    """
    x = tf.layers.conv2d(model_input, 128, 3, padding="same", activation=None, name="conv1")
    y = x

    with tf.variable_scope("rnn"):
        for i in range(state_num):
            reuse = i > 0
            if i == 0:
                x, corr = _residual_block(
                    x, y, 128, is_training, batch_size, key_dim, value_dim,
                    num_heads, name="RB1", reuse=False,
                )
            else:
                x, corr = _residual_block(
                    x, y, 128, is_training, batch_size, key_dim, value_dim,
                    num_heads, name="RB1", reuse=True, corr=corr,
                )

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 1, 3, padding="same", activation=None, name="conv_end")
    return x


def _residual_block(x, y, filter_num, is_training, batch_size,
                    key_dim, value_dim, num_heads, name, reuse, corr=None):
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x, corr_new = _non_local_block(
        x, 128, 128, batch_size, key_dim, value_dim, num_heads,
        name="non_local", reuse=reuse, corr=corr,
    )

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, filter_num, 3, padding="same", activation=None,
                         name=name + "_a", reuse=reuse)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filter_num, 3, padding="same", activation=None,
                         name=name + "_b", reuse=reuse)

    x = tf.add(x, y)
    return x, corr_new


def _non_local_block(x, filter_num, output_filter_num, batch_size,
                     key_dim, value_dim, num_heads, name, reuse=False, corr=None):
    x_theta = tf.layers.conv2d(x, filter_num, 1, padding="same", activation=None,
                                name=name + "_theta", reuse=reuse)
    x_phi = tf.layers.conv2d(x, filter_num, 1, padding="same", activation=None,
                              name=name + "_phi", reuse=reuse)
    x_g = tf.layers.conv2d(x, output_filter_num, 1, padding="same", activation=None,
                            name=name + "_g", reuse=reuse,
                            kernel_initializer=tf.zeros_initializer())

    x_theta_r = tf.reshape(x_theta, [tf.shape(x_theta)[0],
                                      tf.shape(x_theta)[1] * tf.shape(x_theta)[2],
                                      tf.shape(x_theta)[3]])
    x_phi_r = tf.reshape(x_phi, [tf.shape(x_phi)[0],
                                  tf.shape(x_phi)[1] * tf.shape(x_phi)[2],
                                  tf.shape(x_phi)[3]])
    x_g_r = tf.reshape(x_g, [tf.shape(x_g)[0],
                              tf.shape(x_g)[1] * tf.shape(x_g)[2],
                              tf.shape(x_g)[3]])

    # Split heads — dimension-wise
    qs, ks, vs = _split_heads(x_theta_r, x_phi_r, x_g_r,
                               batch_size, key_dim, value_dim, num_heads)
    outputs = _scaled_dot_product(qs, ks, vs, key_dim, num_heads)
    output_dim = _concat_heads(outputs, batch_size)

    # Split heads — sequence-wise
    qs_s, ks_s, vs_s = _split_heads_seq(x_theta_r, x_phi_r, x_g_r,
                                         batch_size, key_dim, value_dim, num_heads)
    outputs_seq = _scaled_dot_product(qs_s, ks_s, vs_s, key_dim, num_heads)
    output_seq = _concat_heads_seq(outputs_seq, batch_size)
    output_seq = tf.reshape(output_seq, [batch_size, -1, filter_num])

    output = tf.concat(values=[output_seq, output_dim], axis=-1)
    x_mul2_reshaped = tf.reshape(
        output,
        [tf.shape(output)[0], tf.shape(x_phi)[1], tf.shape(x_phi)[2],
         output_filter_num * 2],
    )
    output_mix = tf.layers.conv2d(x_mul2_reshaped, filter_num, 1, padding="same",
                                   activation=None, name=name + "_mix_output",
                                   reuse=reuse)

    x_mul1 = output_mix[-1]
    if corr is not None:
        x_mul1 += corr

    return tf.add(x, output_mix), x_mul1


def _split_heads(q, k, v, batch_size, key_dim, value_dim, num_heads):
    def split_last(tensor, nh, dim):
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if s is None else s for s in t_shape]
        tensor = tf.reshape(tensor, [batch_size] + t_shape[1:-1] + [nh, dim // nh])
        return tf.transpose(tensor, [0, 2, 1, 3])

    return (split_last(q, num_heads, key_dim),
            split_last(k, num_heads, key_dim),
            split_last(v, num_heads, value_dim))


def _split_heads_seq(q, k, v, batch_size, key_dim, value_dim, num_heads):
    def split_second(tensor, nh, dim):
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if s is None else s for s in t_shape]
        tensor = tf.reshape(tensor, [batch_size] + [nh, t_shape[1] // nh] + [dim])
        return tf.transpose(tensor, [0, 1, 3, 2])

    return (split_second(q, num_heads, key_dim),
            split_second(k, num_heads, key_dim),
            split_second(v, num_heads, value_dim))


def _scaled_dot_product(qs, ks, vs, key_dim, num_heads):
    key_dim_per_head = key_dim // num_heads
    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (key_dim_per_head ** 0.5)
    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)


def _concat_heads(outputs, batch_size):
    tensor = tf.transpose(outputs, [0, 2, 1, 3])
    t_shape = tensor.get_shape().as_list()
    t_shape = [-1 if s is None else s for s in t_shape]
    num_heads, dim = t_shape[2:]
    return tf.reshape(tensor, [batch_size] + [t_shape[1]] + [num_heads * dim])


def _concat_heads_seq(outputs, batch_size):
    tensor = tf.transpose(outputs, [0, 2, 1, 3])
    t_shape = tensor.get_shape().as_list()
    t_shape = [-1 if s is None else s for s in t_shape]
    return tf.reshape(tensor, [batch_size] + [t_shape[1]] + [-1])
