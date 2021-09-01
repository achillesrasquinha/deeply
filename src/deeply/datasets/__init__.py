import tensorflow_datasets as tfds

from deeply.datasets.jsrt           import JSRT
from deeply.datasets.montgomery     import Montgomery
from deeply.datasets.shenzhen       import Shenzhen
from deeply.datasets.v7darwin       import V7Darwin
from deeply.datasets.siim_covid19   import SiimCovid19
from deeply.datasets.hyper_kvasir.labeled   import HyperKvasirLabeled
from deeply.datasets.hyper_kvasir.segmented import HyperKvasirSegmented

load = tfds.load