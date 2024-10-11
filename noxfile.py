from pathlib import Path

import nox


@nox.session(reuse_venv=True)
def flake8(session):
    session.install(
        "flake8",
        "flake8-absolute-import",
        "flake8-bugbear",
        "flake8-builtins",
        "flake8-colors",
        "flake8-commas",
        "flake8-comprehensions",
        # "flake8-docstrings",
        "flake8-pyproject",
        "flake8-use-fstring",
        "pep8-naming",
    )
    session.run("flake8", "Heimdall/", "train.py")


@nox.session(reuse_venv=True)
def lint(session):
    targets = (flake8,)
    for t in targets:
        session.log(f"Running {t.__name__}")
        t(session)


@nox.session
def unittests(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "requirements_dev.txt")
    session.install("-e", ".")
    session.run("pytest")


@nox.session
def test_experiments(session):
    # Set up vars
    homedir = Path(__file__).resolve().parent
    exp_config_dir = homedir / "config" / "experiments"
    experiments = [i.stem for i in exp_config_dir.glob("*.yaml")]

    small_experiments = [
        "cta_pancreas",
        "pretrain_geneformer_dev",
    ]
    large_experiments = [
        "cta_amb",
        "cta_mpi",
        "cell_cell_interaction_full",
    ]

    # Set up args
    quick_run = full_run = False
    if "full_run" in session.posargs:
        # $ nox -e test_experiments -- full_run
        assert not quick_run
        full_run = True
    if "quick_run" in session.posargs:
        # $ nox -e test_experiments -- quick_run
        assert not full_run
        quick_run = True

    user = "lane-shared-dev"
    if user := [i for i in session.posargs if i.startswith("user=")]:
        # $ nox -e test_experiments -- user=box-remy-dev
        assert len(user) == 1, "Multiple user options not allowed."
        user = user[0].replace("user=", "")

    # Install env
    session.install("-r", "requirements.txt")
    session.install(
        "torch==2.0.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
        # "https://download.pytorch.org/whl/cpu",
    )

    # Run tests
    for exp in experiments:
        if quick_run and exp not in small_experiments:
            continue
        elif not full_run and exp in large_experiments:
            session.log(f"Skipping large experiment {exp!r}")
            continue

        session.log(f"Running experiment {exp!r}")
        session.run(
            "python",
            "train.py",
            f"+experiments={exp}",
            f"user={user}",
            "cache_preprocessed_dataset_dir=null",
        )


nox.options.sessions = [
    "lint",
    "unittests",
]
