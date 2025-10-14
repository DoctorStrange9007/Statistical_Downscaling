import h5py
import jax.numpy as jnp
import tensorflow as tf


def get_raw_datasets(file_name="data/ks_trajectories_512.h5", ds_x=4):
    """Load KS trajectory datasets from an HDF5 file and derive a downsampled field.

    This reads the low-fidelity, low-resolution (`LFLR`), high-fidelity, high-resolution (`HFHR`),
    time (`t`), and space (`x`) arrays from the HDF5 file. It also constructs a
    high-fidelity, low-resolution array (`HFLR`) by downsampling `HFHR` along the spatial axis
    by a stride of `ds_x`.

    Args:
        file_name: Path to an HDF5 file containing datasets 'LFLR', 'HFHR', 't', and 'x'.
        ds_x: Positive integer stride used to downsample the spatial axis of `HFHR` to form `HFLR`.

    Returns:
        Tuple `(u_HFHR, u_LFLR, u_HFLR, x, t)` where each element is a NumPy array.
        `u_HFLR` is computed as `u_HFHR[:, :, ::ds_x]`.

    Raises:
        FileNotFoundError: If the HDF5 file cannot be found.
        KeyError: If required datasets are missing from the file.
    """
    with h5py.File(file_name, "r+") as f1:
        u_LFLR = f1["LFLR"][()]
        u_HFHR = f1["HFHR"][()]
        t = f1["t"][()]
        x = f1["x"][()]

    u_HFLR = u_HFHR[:, :, ::ds_x]
    return u_HFHR, u_LFLR, u_HFLR, x, t


def get_ks_dataset(u_samples: jnp.ndarray, split: str, batch_size: int):
    """Create an infinite, batched NumPy iterator over samples for training.

    Args:
        u_samples: Array-like of samples; each element is exposed under key 'x'.
        split: One of 'train', 'train[:p%]', or 'train[p%:]' to select a prefix or suffix.
        batch_size: Number of examples per batch.

    Returns:
        A repeating, prefetching NumPy iterator yielding batches: {'x': array}.

    Raises:
        ValueError: If `split` is not one of the supported formats.
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
