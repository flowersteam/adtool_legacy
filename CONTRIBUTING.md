# How to Contribute

We are happy to accept contributions to this project. Please follow the
instructions below.

## Branching and Commits

We follow [Gitlab
Flow](https://about.gitlab.com/topics/version-control/what-is-gitlab-flow/),
where short-lived feature branches are used which are frequently merged into the
`dev` branch. The `dev` branch is then manually merged to `prod` at release
milestones.

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) is used
to format commit messages. This allows the Github Actions workflow to
automatically push release tags to the `dev` branch. Merges to the `prod` branch
are done manually on sufficiently stable releases.

## Code, Pullreqs, Merging

All submissions will require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

We use Black + isort to format all Python code and Prettier on all file formats
it supports by default (e.g., JS/HTML/CSS).

We prefer doing a squash merge if the PR is sufficiently small.

## Documentation

Our documentation is automatically generated using pdoc and deployed on the
`gh-pages` branch of this repository (see
[`build_docs.yml`](.github/workflows/build_docs.yml)). You can modify the
README's in the documentation by modifying the Markdown files in `docs`. You
can modify the docstrings for individual classes and methods in the
corresponding Python source files. Create a pull request as usual with your
modifications.
