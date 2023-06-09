#!/usr/bin/env python3
import argparse
import os
import subprocess
from enum import Enum
from typing import List


class HasDockerCompose(Enum):
    """An enum that checks for a docker compose command to run, listed in
    fallback order."""

    HAS_DOCKER_COMPOSE = 1
    HAS_DOCKER_HYPHEN_COMPOSE = 2
    HAS_NONE = 3


def check_docker_compose() -> HasDockerCompose:
    """Check if Docker Compose is installed and can be run without sudo."""
    try:
        subprocess.run(
            ["docker", "compose", "--version"], check=True, stdout=subprocess.DEVNULL
        )
        return HasDockerCompose.HAS_DOCKER_COMPOSE
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        subprocess.run(
            ["docker-compose", "--version"], check=True, stdout=subprocess.DEVNULL
        )
        return HasDockerCompose.HAS_DOCKER_HYPHEN_COMPOSE
    except (FileNotFoundError, subprocess.CalledProcessError):
        return HasDockerCompose.HAS_NONE


def set_notebooks_permissions() -> None:
    """Set permissions for the notebooks directory to resolve permission issues."""
    subprocess.run(["chmod", "-R", "777", "./services/base/JupyterLab/Notebooks/"])


def is_git_repository() -> bool:
    """Check if the current directory is a Git repository."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_git_remote_url() -> str:
    """Get the remote URL of the Git repository."""
    try:
        output = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode()
            .strip()
        )
        return output
    except subprocess.CalledProcessError:
        return ""


def is_correct_git_repository(repo_name: str) -> bool:
    """Check if the current directory corresponds to the expected Git repository.

    Args:
        repo_name (str): The expected name of the Git repository.

    Returns:
        bool: True if the current directory corresponds to the expected Git repository, False otherwise.
    """
    remote_url = get_git_remote_url()
    return repo_name in remote_url


def is_at_root_folder() -> bool:
    """Check if the current directory is at the project root."""
    git_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    cwd = os.getcwd()

    return git_root == cwd


def main() -> None:
    """Entry point for the Dockerized application."""
    repo_name = "AutomatedDiscoveryTool"

    # parse arguments
    parser = argparse.ArgumentParser(description="Entry point for adtool")
    parser.add_argument(
        "--mode",
        choices=["base", "dev", "prod"],
        default="base",
        help="specify the mode: base, dev, or prod (default: base)",
    )
    parser.add_argument(
        "--env",
        metavar="FILE_PATH",
        help="specify the relative file path to source environment variables",
    )
    parser.add_argument("--gpu", action="store_true", help="run with GPU")
    parser.add_argument(
        "command",
        nargs="?",
        default="up",
        help="specify the command for docker-compose",
    )
    args = parser.parse_args()

    # run checks
    if not is_git_repository() or not is_correct_git_repository(repo_name):
        print("Error: start_app.py must be run from the correct Git repository.")
        return
    has_docker_compose = check_docker_compose()
    if has_docker_compose == HasDockerCompose.HAS_NONE:
        print("Error: Docker Compose is not installed or cannot be run without sudo.")
        return
    if not is_at_root_folder():
        print("Error: start_app.py must be run from the root of the project.")
        return

    # HACK: need to ensure notebook permissions which are sometimes wrong
    set_notebooks_permissions()

    # run docker compose
    if args.mode == "dev":
        os.chdir("./services/dev")
    elif args.mode == "prod":
        os.chdir("./services/prod")
    elif args.mode == "base":
        os.chdir("./services/base")

    passed_args = []
    if args.env:
        passed_args.extend(["--env-file", args.env])
    compose_files: List[str] = ["-f", "docker-compose.yml"]
    if args.gpu:
        docker_compose_override_path = "docker-compose-gpu.override.yml"
        # except for the base config, in the others we must go up in the
        # directory to find the override
        if args.mode != "base":
            docker_compose_override_path = "../base/" + docker_compose_override_path

        compose_files.extend(["-f", docker_compose_override_path])

    executable: List[str] = []
    if has_docker_compose == HasDockerCompose.HAS_DOCKER_COMPOSE:
        executable = ["docker", "compose"]
    elif has_docker_compose == HasDockerCompose.HAS_DOCKER_HYPHEN_COMPOSE:
        executable = ["docker-compose"]

    command: List[str] = executable + passed_args + compose_files + [args.command]

    # spawn child process that takes over the python process
    # to handle keyboard interrupts like Ctrl-C
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
