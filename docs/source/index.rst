🧠 deeply
========

Release v\ |version|. (:ref:`Installation <install:>`)

.. image:: https://img.shields.io/pypi/pyversions/ccapi.svg?style=flat-square
    :target: https://pypi.org/project/ccapi/

.. image:: https://img.shields.io/docker/build/achillesrasquinha/ccapi.svg?style=flat-square&logo=docker
    :target: https://hub.docker.com/r/achillesrasquinha/ccapi

.. image:: https://img.shields.io/badge/made%20with-boilpy-red.svg?style=flat-square
    :target: https://git.io/boilpy

.. image:: https://img.shields.io/badge/donate-💵-f44336.svg?style=flat-square
    :target: https://paypal.me/achillesrasquinha

**deeply** is a simple and elegant Deep Learning library written in Python containing a growing collection of deep learning models, datasets and utilities.

----------

**Behold, the power of deeply**:

    >>> # import deeply
    >>> import deeply
    >>> import deeply.datasets as dd
    >>> # load data
    >>> mnist = dd.load("mnist")
    >>> train, val, test = dd.util.split(mnist)
    >>> # build model
    >>> model = deeply.model("efficient-net-b7")
    >>> model.fit(train, validation_data = val, epochs = 50)

⭐ features
-----------

deeply officially supports Python 3.5+.

📚 guides
---------

.. toctree::
   :maxdepth: 2

   models/index
   datasets/index
   ops/index

🤝 contribution
---------------

If you want to contribute to the project, this part of the documentation is for you.