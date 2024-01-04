from setuptools import setup, find_packages

setup(name="gpsampler", packages=find_packages(), install_requires=['gpytorch==1.5.1',
'joblib==1.0.0',
'matplotlib==3.7.0',
'nptyping==2.4.0',
'numba==0.58.1',
'numpy==1.23.5',
'pandas==1.4.0',
'pytest==6.2.2',
'pytest_mock==3.11.1',
'scikit_learn==0.24.1',
'scipy==1.6.0',
'seaborn==0.11.1',
'setuptools==69.0.2',
'statsmodels==0.12.2',
'torch==1.11.0'])