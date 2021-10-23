from functools import partial

import tensorflow_datasets as tfds

from deeply.datasets.jsrt           import JSRT
from deeply.datasets.montgomery     import Montgomery
from deeply.datasets.shenzhen       import Shenzhen
from deeply.datasets.v7darwin       import V7Darwin
from deeply.datasets.siim_covid19   import SiimCovid19
from deeply.datasets.hyper_kvasir.labeled   import HyperKvasirLabeled
from deeply.datasets.hyper_kvasir.segmented import HyperKvasirSegmented
from deeply.datasets.kvasir.segmented import KvasirSegmented

from deeply.datasets.colonoscopy import CVCClinicDB

from bpyutils.util.array import squash

def load(*names, **kwargs):
    results = []
    
    # with parallel.no_daemon_pool(processes = CPU_COUNT) as pool:
    #     results = pool.lmap(
    #         partial(
    #             _load,
    #             **kwargs
    #         )
    #     , names)
    # return results

    for name in names:
        result = tfds.load(name, **kwargs)
        results.append(result)
    
    return squash(results)