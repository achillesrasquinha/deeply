import tensorflow_datasets as tfds

from deeply.datasets.jsrt       import JSRT
from deeply.datasets.montgomery import Montgomery
from deeply.datasets.shenzhen   import Shenzhen
from deeply.datasets.v7darwin   import V7Darwin

load = tfds.load
