"""Commits tasks for pyinvoke."""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task
def serve(ctx: Context) -> None:
    """Run the serving endpoint."""
    ctx.run("uv run python serving_endpoint/server.py", pty=True)


@task(pre=[serve], default=True)
def all(_: Context) -> None:
    """Run all commit tasks."""
