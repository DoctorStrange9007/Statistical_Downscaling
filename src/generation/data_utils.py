import h5py
import jax.numpy as jnp
import tensorflow as tf
from typing import Optional


def get_raw_datasets(file_name, ds_x=4):
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


def get_ks_dataset(
    u_samples: jnp.ndarray, split: str, batch_size: int, seed: Optional[int] = None
):
    """Create a seeded random, infinite, batched NumPy iterator of KS samples.

    The pipeline:
    - casts `u_samples` to float32,
    - optionally subsets via `split`,
    - repeats indefinitely,
    - applies a deterministic circular shift to each element using stateless RNG
      keyed by `(seed, sample_index % len(u_samples))`,
    - batches and prefetches, returning a NumPy iterator of dicts {'x': array}.

    Args:
        u_samples: Array-like of shape (N, L, 1) containing KS fields. Cast to float32.
        split: 'train', 'train[:p%]' to take a prefix, or 'train[p%:]' to take a suffix.
        batch_size: Number of examples per batch.
        seed: Base RNG seed for stateless, per-sample circular shifts. Must be provided
            to ensure reproducible augmentation.

    Returns:
        An endless NumPy iterator yielding dictionaries with key 'x' and value of
        shape (batch_size, L, 1).

    Determinism:
        Given the same `u_samples`, `split`, `batch_size`, and `seed`, the iterator
        yields identical batches across runs. Each element receives a distinct shift
        drawn uniformly from [0, L), fixed by its index.

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

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    ds = ds.repeat()
    global_seed = tf.cast(int(seed), tf.int32)
    ds = ds.enumerate()

    def _seeded_random_roll_map_fn(index, data_dict):
        sample = data_dict["x"]
        sample_len = tf.shape(sample)[0]
        idx_mod = tf.math.floormod(index, tf.cast(total_len, tf.int64))
        idx_mod_i32 = tf.cast(idx_mod, tf.int32)
        shift = tf.random.stateless_uniform(
            shape=[],
            minval=0,
            maxval=sample_len,
            dtype=tf.int32,
            seed=tf.stack([global_seed, idx_mod_i32]),
        )
        rolled_sample = tf.roll(sample, shift=shift, axis=0)
        return {"x": rolled_sample}

    ds = ds.map(_seeded_random_roll_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return ds
