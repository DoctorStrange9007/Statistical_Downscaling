"""Weights & Biases adapter that mirrors `clu.metric_writers` interface.

This module provides `WandbWriter`, a thin wrapper that forwards writes to a
base metric writer (e.g., `clu.metric_writers.MultiWriter`) while also logging
to W&B when available and enabled.
"""

import os

try:
    import wandb  # type: ignore
except Exception:  # wandb is optional
    wandb = None  # type: ignore


class WandbWriter:
    """Adapter that mirrors `clu.metric_writers` to Weights & Biases.

    Forwards writes to a provided base writer and additionally logs to W&B
    when the library is available and not disabled via the environment.
    """

    def __init__(
        self, base_writer, *, project: str, name: str, entity: str = None, config=None
    ):
        """Initialize the adapter.

        Args:
            base_writer: Underlying writer implementing `write_scalars`, etc.
            project: W&B project name.
            name: W&B run name.
            entity: Optional W&B entity (team or user).
            config: Optional configuration dict to attach to the run.
        """
        self.base_writer = base_writer
        self._step = 0
        self._active = bool(wandb is not None and not os.environ.get("WANDB_DISABLED"))
        self._run = None
        if self._active:
            try:
                self._run = wandb.init(
                    project=project,
                    name=name,
                    entity=entity,
                    config=config or {},
                    reinit=True,
                )
            except Exception:
                self._active = False

    def __getattr__(self, item):
        """Proxy unknown attributes to the underlying base writer."""
        return getattr(self.base_writer, item)

    def set_step(self, step: int):
        """Set the current global step for both base writer and W&B."""
        self._step = int(step)
        try:
            if hasattr(self.base_writer, "set_step"):
                self.base_writer.set_step(step)
        except Exception:
            pass

    def write_scalars(self, *args, **kwargs):
        """Write a dictionary of scalar metrics, and mirror to W&B.

        Accepts either positional form `(step, scalars)` or keyword form
        `write_scalars(step=..., scalars={...})`. If `step` is omitted, the
        last set step is used.
        """
        step = kwargs.pop("step", None)
        scalars = kwargs.pop("scalars", None)
        if scalars is None and args:
            if len(args) == 1:
                scalars = args[0]
            elif len(args) >= 2:
                step = args[0]
                scalars = args[1]
        if step is None:
            step = self._step

        try:
            self.base_writer.write_scalars(step=step, scalars=scalars)
        except TypeError:
            try:
                self.base_writer.write_scalars(scalars, step)
            except Exception:
                pass
        except Exception:
            pass

        if self._active and scalars:
            try:
                wandb.log(dict(scalars), step=int(step) if step is not None else None)
            except Exception:
                pass

    def write_hparams(self, hparams):
        """Write hyperparameters to the base writer and W&B config."""
        try:
            if hasattr(self.base_writer, "write_hparams"):
                self.base_writer.write_hparams(hparams)
        except Exception:
            pass
        if self._active and hparams:
            try:
                wandb.config.update(dict(hparams), allow_val_change=True)
            except Exception:
                pass

    def write_images(self, *args, **kwargs):
        """Write images via base writer and mirror to W&B if active.

        Expects keyword argument `images` as a mapping from name to image array.
        """
        try:
            if hasattr(self.base_writer, "write_images"):
                self.base_writer.write_images(*args, **kwargs)
        except Exception:
            pass
        images = kwargs.get("images")
        if self._active and images:
            try:
                wandb.log(
                    {k: wandb.Image(v) for k, v in images.items()}, step=self._step
                )
            except Exception:
                pass

    def flush(self):
        """Flush the base writer buffers if supported."""
        try:
            if hasattr(self.base_writer, "flush"):
                self.base_writer.flush()
        except Exception:
            pass

    def close(self):
        """Close the base writer and finish the W&B run if active."""
        try:
            if hasattr(self.base_writer, "close"):
                self.base_writer.close()
        except Exception:
            pass
        if self._active:
            try:
                wandb.finish()
            except Exception:
                pass
