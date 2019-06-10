#### Setup help:
Make sure you have git installed.
Open terminal, go to wherever you want your local repository, and type:
```
git init
git pull git@github.com:EmoryMLIP/DynamicBlocks.git
```
For running on a linux server, you may want to change the Git default editor:
```
git config --global core.editor "vim"
```

All the code should follow python3, so make sure that you use that throughout. (I get weird errors about the import statements if I try to run anything using python 2.)
You may wish to set up a virtual environment (if not installed, install virtualenv). I called mine torchEnv.
```
python3 -m virtualenv torchEnv  
```

(If this line gives you trouble, see \*Troubleshooting.)


To activate the virtual environment:
```
source torchEnv/bin/activate
```

I did a pip freeze to get all the requirements, removed the hazardous dependencies, and stored them in requirements.txt
To load them into your virtual environment, run:
```
pip3 install -r requirements.txt 
```

Now run the default network (RK4 scheme using a Double Symmetric Layer on the CIFAR-10 dataset):
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
