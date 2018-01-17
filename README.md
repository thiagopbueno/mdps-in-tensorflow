# tf-mdp

Representation and solution of continuous state-action MDPs in TensorFlow.

[![Build Status](https://travis-ci.org/thiagopbueno/tf-mdp.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-mdp)
[![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE)

## Usage

```shell
$ python3 tf_mdp/main.py --help
```

```shell
$ python3 tf_mdp/main.py --mode det -m navigation-small-1-zone -e 100 -lr 0.1
```

```shell
$ python3 tf_mdp/main.py --mode st -m noisy-navigation-small-1-zone -e 100 -lr 0.001
```

```shell
$ python3 tf_mdp/main.py --mode pg -m noisy-navigation-small-1-zone -e 100 -lr 0.001
```

```shell
$ python3 tf_mdp/main.py --mode st -m noisy-navigation-small -e 200 -lr 0.001 -t 8

$ python3 tf_mdp/main.py --mode pg -m noisy-navigation-small -e 100 -lr 0.001 -t 8
