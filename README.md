# Experimentation and Evaluation in Machine Learning

## General

**Author**: Lowejatan Noori

**E-Mail**: l.noori@uni-kassel.de

**Institute**: Intelligent Embedded Systems, University of Kassel, Germany

This repository was created as part of the course *Experimentation and 
Evaluation in Machine Learning*.

## Project Structure

- `e2ml`: Python package of the Python modules implemented during this course
    - `evaluation`: Python package to evaluate and visualize experimental results
    - `experimentation`: Python package with methods of design of experiments
    - `models`: Python package of implement machine learning models
    - `preprocessing`: Python package of data preprocessing functions
- `13_capstone_exercise_summer_term_25.ipynb': Final project with a comparison of DOE methods
- `LICENSE`: information about the terms under which one can use this package
- `setup.py`: Python file to install the project's package

## Setup

To install and use this project, one needs to consider the following steps.

1. Update the general section of `README.md` and the `setup.py` file by adding your credentials to the designated
text passages.
2. Install conda for Python 3.9 according to the 
   [installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Create a conda environment with the name `e2ml-env`.
```shell
conda create --name e2ml-env python=3.9
```
4. Activate the created environment.
```shell
conda activate e2ml-env
```
5. Install the project's package `e2ml` in the conda environment.
```shell
pip install -e .
```
6. Now, you have installed the Python package `e2ml` and should
be able to use it. You can test it by importing it within a Python console.
```python
import e2ml
```
7. Finally, you can start to work with this project. In particular, you can view the 
   [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) in the folder `notebooks`
   by executing the following command.
```shell
jupyter-notebook
```
