<div align="center">
  <img src=".github/assets/logo.png" height="128">
  <h1>
<<<<<<< HEAD
    deeply
  </h1>
</div>

<p align="center">
    <a href='https://github.com/achillesrasquinha/deeply//actions?query=workflow:"Continuous Integration"'>
      <img src="https://img.shields.io/github/workflow/status/achillesrasquinha/deeply/Continuous Integration?style=flat-square">
    </a>
    <a href="https://coveralls.io/github/achillesrasquinha/deeply">
      <img src="https://img.shields.io/coveralls/github/achillesrasquinha/deeply.svg?style=flat-square">
=======
      deeply
  </h1>
  <h4></h4>
</div>

<p align="center">
    <a href='https://github.com//deeply//actions?query=workflow:"Continuous Integration"'>
      <img src="https://img.shields.io/github/workflow/status//deeply/Continuous Integration?style=flat-square">
    </a>
    <a href="https://coveralls.io/github//deeply">
      <img src="https://img.shields.io/coveralls/github//deeply.svg?style=flat-square">
>>>>>>> template/master
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

<<<<<<< HEAD
🧠 deeply is a simple and elegant Deep Learning library written in Python containing a growing collection of deep learning models, datasets and utilities.

**Behold, the power of deeply**:

```python
>>> import deeply
>>> import deeply.datasets as dd
>>> # load mnist
>>> mnist = dd.load("mnist")
>>> (train, val), test = dd.split(mnist["train"], splits = (.8, .2)), mnist["test"]
>>> # build model
>>> model = deeply.hub("efficient-net-b7", pretrained = True)
>>> model.fit(train, validation_data = val, epochs = 10)
```

=======
>>>>>>> template/master
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

<<<<<<< HEAD
Check out [installation](docs/source/installation.md) for more details.
=======
Check out [installation](docs/source/install.rst) for more details.
>>>>>>> template/master

### Usage

#### Application Interface

```python
>>> import deeply
```


#### Command-Line Interface

```console
$ deeply
Usage: deeply [OPTIONS] COMMAND [ARGS]...

<<<<<<< HEAD
  A Deep Learning library
=======
  
>>>>>>> template/master

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  help     Show this message and exit.
  version  Show version and exit.
```


<<<<<<< HEAD
=======
### Docker

Using `deeply's` Docker Image can be done as follows:

```
$ docker run \
    --rm \
    -it \
    /deeply \
      --verbose
```

>>>>>>> template/master
### License

This repository has been released under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ using <a href="https://git.io/boilpy">boilpy</a>.
</div>