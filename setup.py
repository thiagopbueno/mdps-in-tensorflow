from setuptools import setup, find_packages

setup(
    name='tf-mdp',
    version='0.1',
    description='Representation and solution of continuous state-action MDPs in TensorFlow',
    packages=find_packages(),
    test_suite="tests",
)