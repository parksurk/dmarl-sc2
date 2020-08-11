# Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system
for installing multiple versions of software packages and their dependencies and
switching easily between them. It works on Linux, OS X and Windows, and was created
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

**NOTE**: There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).

|        | Linux | Mac | Windows |
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work.


If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
	[Git installation from below]
	-	Linux: https://www.atlassian.com/git/tutorials/install-git#linux
	-	Mac: https://www.atlassian.com/git/tutorials/install-git#mac-os-x
	-	Windows: https://www.atlassian.com/git/tutorials/install-git#windows

	[Git usage reference]
	-	[Reference #1 : Intorduction to Git for Data Science](https://www.datacamp.com/courses/introduction-to-git-for-data-science)
	-	[Reference #2 : Git the simple guide](https://rogerdudler.github.io/git-guide/index.ko.html)

	```
	git clone https://github.com/parksurk/dmarl-sc2.git
	cd dmarl-sc2
	```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__:
	```
	conda create -n starcraft2 python=3.6
	source activate starcraft2
	```
	- __Windows__:
	```
	conda create --name starcraft2 python=3.6
	activate starcraft2
	```

	At this point your command line should look something like: `(starcraft2) <User>:dmarl-sc2 <user>$`. The `(starcraft2)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	-	Please, refer to the PyTorch Installation Guide : https://pytorch.org/get-started/locally/

	- __Linux__ or __Mac__:
	```
	conda install pytorch torchvision -c pytorch
	```
	- __Windows__:
	```
	conda install pytorch torchvision cpuonly -c pytorch
	```
4.	Install TensorFlow; this should install the latest version of TensorFlow.

	-	Please, refer to the TensorFlow Installation Guide : https://www.tensorflow.org/install

	-	**Linux** or **Mac**:
	```
	pip install tensorflow
	```

	-	**Windows**:
	```
	pip install tensorflow
	```

5. Install a few required pip packages, which are specified in the requirements text file (including Pandas, Jupyter, ...).
	```
	pip install pandas
	pip install jupyter
	...
	```

6. Create an IPython kernel for the 'starcraft2' environment.

	```
	pip install ipykernel
	python -m ipykernel install --user --name starcraft2 --display-name "starcraft2"
	```

7. That's it! Now all of the `cv-nd` libraries are available to you. Assuming you're environment is still activated, you can navigate to the Exercises repo and start looking at the notebooks:

	```
	cd
	cd dmarl-sc2
	jupyter notebook
	```

To exit the environment when you have completed your work session, simply close the terminal window.


### Notes on environment creation and deletion

**Verify** that the `starcraft2` environment was created in your environments:

```
conda info --envs
```

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```
conda clean -tp
```

**Uninstall** the environment (if you want); you can remove it by name:

```
conda env remove -n starcraft2
```
