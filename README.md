# Hackathon PMS 2026 Game Packing

The goal is to pack boxes from various board games (data comes from https://www.espritjeu.com) in a rectangular container box with minimum sum of length, width and height. Boxes may be rotated but their axis must remain orthogonal to the axis of the container box.

The folder `dataset` contains 21 instance files. Each CSV file is a set of boxes, the name of the instance indicates the generation method (homogeneous, heterogeneous, random). There is one row per box and the fields are the name, length, width and height of that box. 

You can solve the problem in any way you want, as long as you follow the [solution submission procedure](#submitting-solutions).
There are helpers scripts for the following toolkits:
- [Tempo (recommended)](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/Tempo/README.md?ref_type=heads) 
- [Python](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/CPMpy/README.md) 
- [Minizinc](https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing/-/blob/main/Minizinc/README.md)

The solutions can be visualized using the Python library `blockviz`...

## scripts  

> pip install git+https://gitlab.laas.fr/roc/titouan-seraud/pms.git
(require Python 3.12)


## Setting up your repository

If you have a github account, go to your home, then click NEW, choose "import a repository", then copy/paste https://gitlab.laas.fr/roc/hackathon-pms-2026-game-packing in the text box and import as a public repository (this may take a few minutes).

Here is the procedure to setup your repository to submit solutions. It basically consists of copying this repository content to your new repository you will work on.
1. Create a new empty git repository online. Important requirement: it must be **public**.
2. Clone this repository.
3. Enter the repository
4. Remove the `.git` file.
5. Initialize a new local git repository.
6. Stage the current files.
7. Create the initial commit.
8. Push it on your repository.
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
TODO: finish this


1. Clone this repo

> git clone https://gitlab.laas.fr/ROC/hackathon-pms-2026-game-packing

2. Create an empty repository on any git platform (say git@github.com/me/A-team)

3. push the team and solution files in your repo

>>>

cd hackathon-pms-2026-game-packing

git remote add team.json git@github.com/me/A-team

git remote add solutions git@github.com/me/A-team

>>>

4. Fill the file `team.json` with the relevant information

5. *When you find a new best solution*, copy it into the corresponding solution file and push into the corrsponding solution file, e.g., if this is a solution for `dataset/random_0005.csv`, copy it into `solutions/random_0005.json` and:

>>>

git commit -m "new solution" solutions/random_0005.json 

git push main

>>>




