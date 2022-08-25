<<<<<<< HEAD
🧠 deeply
=========

Release v\ |version|. (:ref:`Installation <install>`)
=======
deeply
===

Release v\ |version|. (:ref:`Installation <install:>`)

.. image:: https://img.shields.io/coveralls/github/achillesrasquinha/ccapi.svg?style=flat-square
    :target: https://coveralls.io/github/achillesrasquinha/ccapi
>>>>>>> template/master

.. image:: https://img.shields.io/pypi/pyversions/ccapi.svg?style=flat-square
    :target: https://pypi.org/project/ccapi/

.. image:: https://img.shields.io/docker/build/achillesrasquinha/ccapi.svg?style=flat-square&logo=docker
    :target: https://hub.docker.com/r/achillesrasquinha/ccapi

.. image:: https://img.shields.io/badge/made%20with-boilpy-red.svg?style=flat-square
    :target: https://git.io/boilpy

<<<<<<< HEAD
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
    >>> (train, val), test = dd.split(mnist["train"], splits = (.8, .2)), mnist["test"]
    >>> # build model
    >>> model = deeply.hub("efficient-net-b7")
    >>> model.fit(train, validation_data = val, epochs = 10)

⭐ features
-----------

- Create end-to-end pipeline repositories using deeply templates.
- Avoid unnecessary code so you can simply focus on product delivery.
- Integrate third-party MLOps Infrastructure for real-time experiment tracking with breeze.
- Access to a wide range of datasets.

deeply officially supports Python 3.5+.

📚 guides
---------

.. toctree::
   :maxdepth: 2

   template/index
   models/index
   datasets/index
   ops/index

api
---

.. toctree::
   :maxdepth: 1

   api/metrics

🤝 contribution
---------------

If you want to contribute to the project, this part of the documentation is for you.
=======
.. image:: https://img.shields.io/badge/Say%20Thanks-🦉-1EAEDB.svg?style=flat-square
    :target: https://saythanks.io/to/achillesrasquinha

.. image:: https://img.shields.io/badge/donate-💵-f44336.svg?style=flat-square
    :target: https://paypal.me/achillesrasquinha

>>>>>>> template/master
