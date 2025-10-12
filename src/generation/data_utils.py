import h5py
import jax.numpy as jnp
import tensorflow as tf


def get_raw_datasets(file_name="data/ks_trajectories_512.h5", ds_x=4):
    with h5py.File(file_name, "r+") as f1:
        u_LFLR = f1["LFLR"][()]
        u_HFHR = f1["HFHR"][()]
        t = f1["t"][()]
        x = f1["x"][()]

    u_HFLR = u_HFHR[:, :, ::ds_x]
    return u_HFHR, u_LFLR, u_HFLR, x, t


def get_ks_dataset(u_samples: jnp.ndarray, split: str, batch_size: int):
    """Returns a batched dataset from u_samples with the same interface as get_mnist_dataset.

    Args:
        u_samples: Array of shape (N, 192, 1)
        split: A TFDS-style split string (e.g., 'train[:75%]')
        batch_size: Batch size for training

    Returns:
        A NumPy iterator over batches of {'x': ...}
    """
    ds = tf.data.Dataset.from_tensor_slices({"x": u_samples.astype(jnp.float32)})

    total_len = len(u_samples)
    if split == "train":
        pass
    elif split.startswith("train[:"):
        frac = float(split[len("train[:") : -2]) / 100
        ds = ds.take(int(frac * total_len))
    elif split.startswith("train["):
        frac = float(split[len("train[") : -3]) / 100
        ds = ds.skip(int(frac * total_len))
    else:
        raise ValueError(f"Unsupported split string: {split}")

    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return ds
