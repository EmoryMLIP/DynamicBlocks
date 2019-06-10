
# PyTorch
PyTorch implementations of Machine learning architectures planning for publication-level comparison with Meganet implementations 

#### Setup help:
Make sure you have git installed.
Open terminal, go to wherever you want your local repository, and type:
```
git init
git pull git@github.com:EmoryMLIP/PyTorch.git
```
For running on Titan or any linux server, you may want to change the Git default editor:
```
git config --global core.editor "vim"
```
(citation: https://stackoverflow.com/questions/2596805/how-do-i-make-git-use-the-editor-of-my-choice-for-commits)

All the code should follow python3, so make sure that you use that throughout. (I get weird errors about the import statements if I try to run anything using python 2.)
You may wish to set up a virtual environment (if not installed, install virtualenv). I called mine torchEnv.
```
python3 -m virtualenv torchEnv  
```

(If this line gives you trouble, see \*Troubleshooting.)


I can never remember how to start the virtual envirnoment, so I included that command in go.
To start things, I type
```
source go
```

I did a pip freeze to get all the requirements, removed the hazardous dependencies, and stored them in requirements.txt
To load them into your virtual environment, run:
```
pip3 install -r requirements.txt 
```

This approach does have some concerns in the community (https://medium.com/@tomagee/pip-freeze-requirements-txt-considered-harmful-f0bce66cf895) mainly because it promotes the continued use of outdated libraries and dependencies become messy.

Now run the default network (RK4 scheme using a doubleSymLayer on the CIFAR-10 dataset):
```
python3 RKNet.py 
```

#### NOTE: default behavior assumes that when running on cpu, the user is debugging. As such, functions will overwrite many parameters to train much smaller models and prevent the cpu from crashing.

#### When running outside of an IDE:

To run python functions in any subfolders, add the root folder to the path via:
```
source startup.sh
```

Then, run the functions from the root folder; for example, run the rk4 test as:

```
python3 modules/testRK4Block.py 
```



-----------------------------------------------------------------------------------------------------------------------
### \*Troubleshooting setup

Check the default python for creating virtual environments 
```
echo $VIRTUALENV_PYTHON
```

You can set this to python3:
```
export VIRTUALENV_PYTHON=/usr/bin/python3
```

Also,  ```virtualenv -p python3 torchEnv``` can be an alternative to ```python3 -m virtualenv torchEnv``` .

-----------------------------------------------------------------------------------------------------------------------
A naming convention I attempt to follow:

nBlah   - integer or 'number of' Blah

fBlah   - float

sBlah   - string

vBlah   - vector

mBlah   - matrix

idxBlah - indices

npBlah  - numpy item

in pytorch, most things are just tensors

------------------------------------------------
variables that appear a lot:

tY     - time points in Y     (these correspond to the state layers)

tTheta - time points in theta (these correspond to the control layers)


