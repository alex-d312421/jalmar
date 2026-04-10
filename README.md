# Hackathon PMS 2026 Game Packing

The goal is to pack boxes from various board games (data comes from https://www.espritjeu.com) in a rectangular container box with minimum sum of length, width and height. Boxes may be rotated but their axis must remain orthogonal to the axis of the container box.

The folder `dataset` contains 21 instance files. Each CSV file is a set of boxes, the name of the instance indicates the generation method (homogeneous, heterogeneous, random). There is one row per box and the fields are the name, length, width and height of that box. 

You can solve the problem in any way you want, as long as you follow the [solution submission procedure](#submitting-solutions).
There are helpers scripts for the following toolkits:
- [Tempo](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/Tempo/README.md?ref_type=heads) 
- [Python](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/CPMpy/README.md) 
- [Minizinc](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/Minizinc/README.md)

## Prerequisites

- Tempo: standard c++20, >= gcc-9 ou >= clang-17

- Visualization: Python 3.12

## scripts  


First install the parsing and visualization scripts (requires Python 3.12):

> pip install git+https://gitlab.laas.fr/roc/titouan-seraud/pms.git

## Pipeline description
TODO


## Repository setup
Here is the procedure to setup your repository to submit solutions. It basically consists of copying this repository content to your new repository you will work on. You have three alternatives for this: GitHub import, GitLab import or manual.

### GitHub import
Sign in and go to your GitHub home, then click *NEW*, choose *Import a repository*, then copy/paste https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing in the text box and import as a **public** repository (this may take a few minutes).

### GitLab import
Sign in and go to your GitLab home. Click on *Projects* on the left pannel, then on *New project* and select *Import project*. Import from *Repository by URL* and copy/paste https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing.git in the text box. Finish to fill-in the form but make sure you select **Public** visibility.

### Manual
The procedure basically copy files from one repository to another. Here are the steps to follow, the commands are underneath.
1. Create a new empty git repository online. Important requirement: it must be **public**.
2. Clone this repository.
3. Enter the repository
4. Remove the `.git` file.
5. Initialize a new local git repository.
6. Link your brand new remote repository.
7. Stage the current files.
8. Create the initial commit.
9. Push it on your remote repository.
```
git clone git@gitlab.laas.fr:roc/hackathon-pms-2026-game-packing.git
cd hackathon-pms-2026-game-packing
rm -rf .git
git init
git remote add origin git@github.com:user/myrepo.git
git add .
git commit -m "Initial commit."
git push --set-upstream origin main
```

## Installation
`pms` and `blockviz` python packages are necessary, regardless of your solving techniques. The first one contains scripts to check and visualize solutions while the second contains the 3D rendering system.

Then, you can choose the framework you prefer, depending on skills and constraint programming knowledge. Here are the three supported frameworks we have tested, from easiest to hardest.
- **MiniZinc** is a modelling language and tool chain for constraint optimisation problems
- **CPMpy** is a Constraint Programming and Modeling library in Python
- **Tempo** is a C++ solver developed by the organizers

### pms and blockviz
First, make sure you have at least python 3.12 installed. Then, make sure your python environment is ready. Once it is, install both `pms` with `pip`.
```
pip install git+https://gitlab.laas.fr/roc/titouan-seraud/pms.git
```
Since `pms` depends on `blockviz`, both of them are installed.

You can check the installation succeded with the following commands.
```
pms-check-repo --help
blockviz-mock | blockviz
```

If you have any issue or want more details on these packages, please read the documentation in their respective repository.
- Blockviz &rarr; [https://gitlab.laas.fr/roc/titouan-seraud/blockviz](https://gitlab.laas.fr/roc/titouan-seraud/blockviz)
- PMS &rarr; [https://gitlab.laas.fr/roc/titouan-seraud/pms](https://gitlab.laas.fr/roc/titouan-seraud/pms)

### MiniZinc
Please follow the [official installation procedure](https://docs.minizinc.dev/en/stable/installation.html).

### CPMpy
Please follow the [official installation procedure](https://cpmpy.readthedocs.io/en/latest/installation_instructions.html).

### Tempo
TODO

## Subscribing your team
TODO

## Useful links
TODO