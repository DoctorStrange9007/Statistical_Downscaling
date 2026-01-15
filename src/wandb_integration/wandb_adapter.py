"""Weights & Biases adapter that mirrors `clu.metric_writers` interface.

This module provides `WandbWriter`, a thin wrapper that forwards writes to a
base metric writer (e.g., `clu.metric_writers.MultiWriter`) while also logging
to W&B.
"""

import wandb  # type: ignore


class WandbWriter:
    """Adapter that mirrors `clu.metric_writers` to Weights & Biases.

    Forwards writes to a provided base writer and additionally logs to W&B
    when the library is available and not disabled via the environment.
    """

    def __init__(
        self,
        base_writer,
        *,
        project: str,
        name: str,
        entity: str = None,
        config=None,
        active: bool = True,
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
        # Initialize a W&B run immediately; assume W&B is available.
        self._run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            config=config or {},
            reinit=True,
        )

    def __getattr__(self, item):
        """Proxy unknown attributes to the underlying base writer."""
        return getattr(self.base_writer, item)

    def set_step(self, step: int):
        """Set the current global step for both base writer and W&B."""
        self._step = int(step)
        if hasattr(self.base_writer, "set_step"):
            self.base_writer.set_step(step)

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
        if not scalars:
            return
        # Allow callers to provide step inside the payload; remove it from metrics
        step_was_provided = step is not None
        if not step_was_provided and isinstance(scalars, dict) and ("step" in scalars):
            step = int(scalars.pop("step"))
            step_was_provided = True
        if step is None:
            step = self._step

        # Simple sanitation: drop None and coerce to Python floats
        sanitized = {}
        for k, v in dict(scalars).items():
            if v is None:
                continue
            try:
                if hasattr(v, "item"):
                    v = v.item()
                sanitized[k] = float(v)
            except Exception:
                # Skip non-convertible values
                continue
        if not sanitized:
            return

        # Forward to base writer
        self.base_writer.write_scalars(step=step, scalars=sanitized)

        # Mirror to W&B
        wandb.log(dict(sanitized), step=step)

        # Advance step only when caller didn't explicitly provide a step
        if not step_was_provided and step >= self._step:
            self._step = step + 1

    def log(self, payload: dict):
        """Convenience: accept a plain dict of scalars and log it."""
        self.write_scalars(scalars=payload)

    def write_scalar(self, name: str, value):
        """Convenience: write a single scalar by name.

        This avoids step management and mirrors directly to W&B when active.
        """
        # Keep W&B step aligned with our tracked step
        wandb.log({name: value}, step=self._step)

    def write_hparams(self, hparams):
        """Write hyperparameters to the base writer and W&B config."""
        if hasattr(self.base_writer, "write_hparams"):
            self.base_writer.write_hparams(hparams)
        if hparams:
            wandb.config.update(dict(hparams), allow_val_change=True)

    def write_images(self, *args, **kwargs):
        """Write images via base writer and mirror to W&B if active.

        Expects keyword argument `images` as a mapping from name to image array.
        """
        images = kwargs.get("images")
        step = kwargs.get("step", self._step)
        if not images:
            return

        if hasattr(self.base_writer, "write_images"):
            array_images = {k: v for k, v in images.items() if hasattr(v, "shape")}
            if array_images:
                try:
                    self.base_writer.write_images(step=step, images=array_images)
                except TypeError:
                    self.base_writer.write_images(step, array_images)
        try:
            wandb.log({k: wandb.Image(v) for k, v in images.items()}, step=step)
        except Exception:

            payload = {}
            for k, v in images.items():
                try:
                    payload[k] = wandb.Image(v)
                except Exception:
                    continue
            if payload:
                wandb.log(payload, step=step)

    def flush(self):
        """Flush the base writer buffers if supported."""
        if hasattr(self.base_writer, "flush"):
            self.base_writer.flush()

    def close(self):
        """Close the base writer and finish the W&B run if active."""
        if hasattr(self.base_writer, "close"):
            self.base_writer.close()
        wandb.finish()
