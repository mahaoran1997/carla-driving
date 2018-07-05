CARLA CLIENT Documentation
===================

This is documentation for the CARLA Python client.

The _CARLA client_ is a python tcp client used to interface with
the _CARLA plugin_ on Unreal Engine 4.17.



Index
-----

#### Setup
Carla Client needs the following dependencies:

* python 2.7 or 3.3 
* protobuf 3.3


Extra Dependencies:

* pygame ( for the keyboard controlling example)


### Basic Usage

Basic Example (carla_use_example.py  < host > <port> )
-pm to print
-lv to have a full log

sample use in a local host

With logs:

python carla_use_example.py 127.0.0.1 2000 -pm -lv

Without logs:

python carla_use_example.py 127.0.0.1 2000


### Other Demos

Basic Keyboard controller ( To be added soon)

HDF5 Data Collector ( To be added)


