"""Install tasks for pyinvoke."""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task
def uv(ctx: Context) -> None:
    """Install uv packages."""
    ctx.run("uv sync --group commits")


@task
def pre_commit(ctx: Context) -> None:
    """Install pre-commit hooks on git."""
    ctx.run("uv run pre-commit install --hook-type pre-push")
    ctx.run("uv run pre-commit install --hook-type commit-msg")


@task(pre=[uv, pre_commit], default=True)
def all(_: Context) -> None:
    """Run all install tasks."""
