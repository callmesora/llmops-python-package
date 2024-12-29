"""Check tasks for pyinvoke."""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task
def uv(ctx: Context) -> None:
    """Check uv config files."""
    ctx.run("uv lock --check", pty=True)


@task
def format(ctx: Context) -> None:
    """Check the formats with ruff."""
    ctx.run("uv run ruff format --check src/ tasks/ tests/", pty=True)


@task
def type(ctx: Context) -> None:
    """Check the types with mypy."""
    ctx.run("uv run mypy src/ tasks/ tests/", pty=True)


@task
def code(ctx: Context) -> None:
    """Check the codes with ruff."""
    ctx.run("uv run ruff check src/ tasks/ tests/", pty=True)


@task
def test(ctx: Context) -> None:
    """Check the tests with pytest."""
    ctx.run(
        "uv run pytest "
        "tests/pipelines/feature_engineering/test_create_vector_db.py "  # Feature Engineering
        "tests/pipelines/feature_engineering/test_ingest_documents.py "  # Feature Engineering
        "tests/pipelines/monitoring/test_generate_rag_dataset.py "  # Monitoring
        "tests/pipelines/deployment/test_register_model.py "  # Deployment
        "tests/pipelines/monitoring/test_pre_deploy_eval.py "  # Monitoring
        "tests/pipelines/deployment/test_deploy_model.py "  # Deployment
        "tests/io/test_services.py "  # IO
        "tests/io/test_configs.py "  # IO
        "tests/pipelines/test_base.py ",  # Base
        # "--numprocesses='auto'"
        pty=True,
    )


@task
def security(ctx: Context) -> None:
    """Check the security with bandit."""
    ctx.run("uv run bandit --recursive --configfile=pyproject.toml src/", pty=True)


@task
def coverage(ctx: Context) -> None:
    """Check the coverage with coverage."""
    ctx.run(
        "uv run pytest --cov=src/ --cov-fail-under=20 "
        "tests/pipelines/feature_engineering/test_create_vector_db.py "  # Feature Engineering
        "tests/pipelines/feature_engineering/test_ingest_documents.py "  # Feature Engineering
        "tests/pipelines/monitoring/test_generate_rag_dataset.py "  # Monitoring
        "tests/pipelines/deployment/test_register_model.py "  # Deployment
        "tests/pipelines/monitoring/test_pre_deploy_eval.py "  # Monitoring
        "tests/pipelines/deployment/test_deploy_model.py "  # Deployment
        "tests/io/test_services.py "  # IO
        "tests/io/test_configs.py "  # IO
        "tests/pipelines/test_base.py ",  # Base
        # "--numprocesses='auto'"
        pty=True,
    )


@task(pre=[uv, format, type, code, security, coverage], default=True)
def all(_: Context) -> None:
    """Run all check tasks."""
