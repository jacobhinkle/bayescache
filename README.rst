.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/bayescache/blob/master/img/pylibrary.png">
        </p>
    </embed>

--------------------------

.. image:: https://badge.fury.io/py/bayescache.png
    :target: http://badge.fury.io/py/bayescache

.. image:: https://travis-ci.org/yngtodd/bayescache.png?branch=master
    :target: https://travis-ci.org/yngtodd/bayescache


==========
BayesCache
==========

Save cache, Bae will you love you.

This project owes a great debt to the `Vel`_ project from `MillionIntegrals`_. I have basically been
pulling apart his API to better understand it. As of right now, this API is nearly identical to that
project. Things may be adapted and diverge from `Vel` as time goes on in order to suit specific needs.

Documentation
-------------
 
For references, tutorials, and examples check out our `documentation`_.

Usage
-----

The command line interface for BayesCache is as follows:

.. code-block:: console

    python -m bayescache.launch CONFIGFILE COMMAND --device PYTORCH_DEVICE -r RUN_NUMBER -s SEED


Installation
------------

From Sources:

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/yngtodd/bayescache

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/yngtodd/bayescache/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python setup.py install

.. _tarball: https://github.com/yngtodd/bayescache/tarball/master
.. _documentation: https://bayescache.readthedocs.io/en/latest
.. _MillionIntegrals: https://github.com/MillionIntegrals
.. _Vel: https://github.com/MillionIntegrals/vel