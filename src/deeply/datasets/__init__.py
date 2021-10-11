from functools import partial

import tensorflow_datasets as tfds

from deeply.datasets.jsrt           import JSRT
from deeply.datasets.montgomery     import Montgomery
from deeply.datasets.shenzhen       import Shenzhen
from deeply.datasets.v7darwin       import V7Darwin
from deeply.datasets.siim_covid19   import SiimCovid19
from deeply.datasets.hyper_kvasir.labeled   import HyperKvasirLabeled
from deeply.datasets.hyper_kvasir.segmented import HyperKvasirSegmented

from bpyutils import parallel
from bpyutils.const import CPU_COUNT

# load = tfds.load
def load(*names, **kwargs):
    with parallel.no_daemon_pool(processes = CPU_COUNT) as pool:
        results = pool.map(partial(tfds.load, **kwargs), names)
        
    return results