# This project is deprecated. Please visit [tf-mdp](https://github.com/thiagopbueno/tf-mdp).


# tf-mdp

Representation and solution of continuous state-action MDPs in TensorFlow.

[![Build Status](https://travis-ci.org/thiagopbueno/tf-mdp.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-mdp)
[![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE)

## Usage

```shell
$ python3 tf_mdp/main.py --help
```

```shell
$ python3 tf_mdp/main.py --mode pg -m navigation-small-no-deceleration -e 100 -lr 0.01 -t 8 --log-dir results/pg/navigation-small-no-deceleration/

$ python3 tf_mdp/main.py --mode pg -m navigation-small-1-zone -e 100 -lr 0.01 -t 8 --log-dir results/pg/navigation-small-1-zone/

$ python3 tf_mdp/main.py --mode pg -m navigation-small-2-zones -e 100 -lr 0.01 -t 15 --log-dir results/pg/navigation-small-2-zones/
```
