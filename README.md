<div align="center">
  <img src=".github/assets/logo.png" height="128">
  <h1>
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

Check out [installation](docs/source/install.rst) for more details.

### Usage

#### Application Interface

```python
>>> import deeply
```


#### Command-Line Interface

```console
$ deeply
Usage: deeply [OPTIONS] COMMAND [ARGS]...

  

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  help     Show this message and exit.
  version  Show version and exit.
```


### Docker

Using `deeply's` Docker Image can be done as follows:

```
$ docker run \
    --rm \
    -it \
    /deeply \
      --verbose
```

### License

This repository has been released under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ using <a href="https://git.io/boilpy">boilpy</a>.
</div>