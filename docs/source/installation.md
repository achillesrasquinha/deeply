### Installation

#### Installation via pip

The recommended way to install **deeply** is via `pip`.

```shell
$ pip install deeply
```

For instructions on installing python and pip see “The Hitchhiker’s Guide to Python” 
[Installation Guides](https://docs.python-guide.org/starting/installation/).

#### Building from source

`deeply` is actively developed on [https://github.com](https://github.com/achillesrasquinha/deeply)
and is always avaliable.

You can clone the base repository with git as follows:

```shell
$ git clone https://github.com/achillesrasquinha/deeply
```

Optionally, you could download the tarball or zipball as follows:

##### For Linux Users

```shell
$ curl -OL https://github.com/achillesrasquinha/tarball/deeply
```

##### For Windows Users

```shell
$ curl -OL https://github.com/achillesrasquinha/zipball/deeply
```

Install necessary dependencies

```shell
$ cd deeply
$ pip install -r requirements.txt
```

Then, go ahead and install deeply in your site-packages as follows:

```shell
$ python setup.py install
```

Check to see if you’ve installed deeply correctly.

```shell
$ deeply --help
```