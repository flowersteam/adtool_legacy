# adtool: Assisted and Automated Discovery for Complex Systems

We're pleased to introduce adtool (formerly, Automated Discovery Tool), a
software for assisted and automated discovery of patterns in the exploration of
complex systems.

adtool is a software package developed in the [Inria
FLOWERS](https://flowers.inria.fr) research team which provides an integrated
solution for studying complex systems through curiosity-search methods,
consisting of a user-friendly Web UI and an extensible Python library for
user-defined experimentation systems and search algorithms.

Searching the configuration space of complex systems is often done manually,
i.e., by a human who individually identifies interesting patterns or behaviors
of the system. adtool thus assists in automating this
_exploratory phase_ of researching a new system which is theorized to be capable
of interesting, yet unknown behavior. This is the case for many projects in the
natural sciences and elsewhere. For example, physicists and chemists may use the
tool study the emergence of novel structures and materials from a physical
system, or digital artists and designers may use the tool to automatically
generate or iterate on existing designs during the creative process.

Please note that this software is currently in an **alpha stage** of
development: it is functional and has been used internally at Inria FLOWERS to
study cellular automata since 2021, but may not have features which are
convenient for different workflows. For more details on the development of
adtool, see the following [usage and technical
section](#usage-and-technical-documentation).

![Short demo](demo.gif)

In the above demo, the tool is used to discover life-like propagating patterns
in a cellular automata simulation, showcasing how a researcher can specify a
series of experiments and monitor their results through both the Web UI and an
integrated Jupyter notebook.

The software was designed and maintained with contributions from Chris Reinke,
Clément Romac, Matthieu Perie, Mayalen Etcheverry, Jesse Lin, and other
collaborators in the FLOWERS team.

## Summary

### Scientific Background: Curiosity Search

The high-dimensional phase space of a complex system poses many challenges to
study. In particular, it is often desirable to explore the behavior space of
such systems for interesting behaviors without knowing a priori the precise
quantities to look for. As such, a class of algorithms based on intrinsic
motivation or "curiosity" has been proposed in [Reinke et al.,
2020](https://arxiv.org/abs/1908.06663) and extended in e.g., [Etcheverry et
al., 2020](https://arxiv.org/abs/2007.01195) ![Lenia](lenia.png) Such curiosity
algorithms enable a system to automatically generate a learning curriculum from
which it learns to explore its behavior space autonomously in search of
interesting behaviors, originally proposed in the context of robotic agents
learning to interact with their environment in an unsupervised manner, as in
[Oudeyer et al., 2007](https://ieeexplore.ieee.org/document/4141061).

In practice, dealing with such ill-posed and/or subjective search tasks requires
significant human oversight. For this reason, our adtool
proposes a software package for both :

- the implementation of such experimental pipelines for arbitrary systems and
  search methods, and
- the human-supervised exploration of such systems.

The repo comes with an existing implementation of the [Lenia
system](https://chakazul.github.io/lenia.html) which can be explored using the
curiosity search algorithms described. The Python API described in the
[technical documentation](https://developmentalsystems.org/adtool/) can be used
to add custom systems and (optionally) search algorithms.

### Installation

The application uses [Docker](https://www.docker.com), which should be installed
manually following the [official
instructions](https://docs.docker.com/engine/install/). The application has been
"real-world tested" on MacOS and Linux, so on Windows, installation under [WSL
(Windows Subsystem for
Linux)](https://learn.microsoft.com/fr-fr/windows/wsl/install) should be
preferred.

To install, simply clone the `prod` branch of the repository, which is where the
stable version is kept, via e.g.,

```bash
git clone https://github.com/flowersteam/adtool.git --branch prod
```

After installation, the appropriate environment variables should be set by
creating a file `services/prod/.env` with the environment variables appropriate
for your setup. A sample is provided at `services/prod/sample.env`

### Basic Usage

Running `start_app.py --mode prod` will launch the Docker containers, from which
the UI can be accessed by default at `http://localhost:4200`

### Developer's Documentation

For development work on the Python libraries, we recommend starting a new conda
environment with

```bash
conda create -n adtool python=3.11 poetry=1.* -y
```

or, if you are running on the new Macbooks with M1 processors

```bash
CONDA_SUBDIR=osx-64 conda create -n adtool python=3.11 poetry=1.* -y
```

Please see the online documentation at https://developmentalsystems.org/adtool/.

Please note that as the software is currently in an **alpha stage** as of
January 2023, breaking changes to the API may occur. This is due to the current
progress towards two key developmental milestones:

- Streamlining of the API, allowing checkpointing and restoring of experiments
  with arbitrary data, à la Git
- Implementation of a human interface for interaction with the search process
  itself, allowing custom human intervention at each step taken

Due to the software currently being in an early stage of development, there may
be questions which are not immediately answered by documentation, and you may
instead direct them to [Jesse Lin](https://github.com/jesseylin) who is happy to
respond.
