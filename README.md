<div align="center">
  <img src=".github/assets/logo.png" height="128">
  <h1>
      deeply
  </h1>
</div>

<p align="center">
    <a href='https://github.com/achillesrasquinha/deeply//actions?query=workflow:"Continuous Integration"'>
      <img src="https://img.shields.io/github/workflow/status/achillesrasquinha/deeply/Continuous Integration?style=flat-square">
    </a>
    <a href="https://coveralls.io/github/achillesrasquinha/deeply">
      <img src="https://img.shields.io/coveralls/github/achillesrasquinha/deeply.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/deeply/">
      <img src="https://img.shields.io/pypi/v/deeply.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/deeply/">
      <img src="https://img.shields.io/pypi/l/deeply.svg?style=flat-square">
    </a>
    <a href="https://pypi.org/project/deeply/">
		  <img src="https://img.shields.io/pypi/pyversions/deeply.svg?style=flat-square">
	  </a>
    <a href="https://git.io/boilpy">
      <img src="https://img.shields.io/badge/made%20with-boilpy-red.svg?style=flat-square">
    </a>
</p>

deeply is a simple and elegant Deep Learning library written in Python containing a growing collection of deep learning models, datasets and utilities.

**Behold, the power of deeply**:

```python
>>> # import deeply
>>> import deeply
>>> import deeply.datasets as dd
>>> # load data
>>> mnist = dd.load("mnist")
>>> (train, val), test = dd.split(mnist["train"], splits = (.8, .2)), mnist["test"]
>>> # build model
>>> model = deeply.model("efficient-net-b7")
>>> model.fit(train, validation_data = val, epochs = 10)
```

### Table of Contents
* [Features](#features)
* [Quick Start](#quick-start)
* [Usage](#usage)
* [License](#license)

### Features
* Python 2.7+ and Python 3.4+ compatible.

### Quick Start

```shell
$ pip install deeply
```

Check out [installation](docs/source/installation.md) for more details.

### Usage

#### Application Interface

```python
>>> import deeply
```


#### Command-Line Interface

```console
$ deeply
Usage: deeply [OPTIONS] COMMAND [ARGS]...

  A Deep Learning library

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  help     Show this message and exit.
  version  Show version and exit.
```


### License

This repository has been released under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ using <a href="https://git.io/boilpy">boilpy</a>.
</div>