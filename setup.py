from setuptools import setup, find_packages

setup(
    name='tf-mdp',
    version='0.1',
    description='Representation and solution of continuous state-action MDPs in TensorFlow',  # noqa
    url='https://github.com/thiagopbueno/tf-mdp',
    license='GNU General Public License',
    author='Thiago Bueno, Felipe Salvatore',
    packages=find_packages(),
    test_suite="tests",
    include_package_data=True,
    platforms='any non-Windows platform',
)
