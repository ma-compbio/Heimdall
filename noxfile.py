import nox


@nox.session
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


@nox.session
def lint(session):
    targets = (flake8,)
    for t in targets:
        session.log(f"Runing {t.__name__}")
        t(session)


@nox.session
def unittests(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "requirements_dev.txt")
    session.install(".")
    session.run("pytest")


@nox.session
def test_experiments(session):
    session.install("-r", "requirements.txt")
    session.install(
        "torch==2.0.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
        # "https://download.pytorch.org/whl/cpu",
    )

    experiments = [
        "cell_cell_interaction_dev",
        "pancreas",
        "pretrain_geneformer_dev",
        "reverse_perturbation",
    ]
    for exp in experiments:
        session.log(f"Runing {exp}")
        session.run("python", "train.py", f"+experiments={exp}", "user=lane-shared-dev")


nox.options.sessions = [
    "lint",
    "unittests",
]
