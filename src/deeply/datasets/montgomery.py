import os.path as osp
from glob import glob

import tensorflow as tf
from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF,
    Text,
    Tensor,
    ClassLabel,
)
import numpy as np
from PIL import Image
import imageio

from deeply.util.string import strip, safe_decode
from deeply.util.system import makedirs
from deeply._compat import iterkeys, iteritems

_DATASET_URL         = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
_DATASET_HOMEPAGE    = "https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html"
_DATASET_DESCRIPTION = """
The MC set has been collected in collaboration with the Department of Health and Human Services, Montgomery County, Maryland, USA. The set contains 138 frontal chest X-rays from Montgomery County’s Tuberculosis screening program, of which 80 are normal cases and 58 are cases with manifestations of TB. The X-rays were captured with a Eureka stationary X-ray machine (CR), and are provided in Portable Network Graphics (PNG) format as 12-bit gray level images. They can also be made available in DICOM format upon request. The size of the X-rays is either 4,020×4,892 or 4,892×4,020 pixels.
"""
_DATASET_CITATION    = """\
@article{jaeger_two_2014,
	title       = {Two public chest {X}-ray datasets for computer-aided screening of pulmonary diseases},
	volume      = {4},
	issn        = {2223-4292},
	url         = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/},
	doi         = {10.3978/j.issn.2223-4292.2014.11.20},
	abstract    = {The U.S. National Library of Medicine has made two datasets of postero-anterior (PA) chest radiographs available to foster research in computer-aided diagnosis of pulmonary diseases with a special focus on pulmonary tuberculosis (TB). The radiographs were acquired from the Department of Health and Human Services, Montgomery County, Maryland, USA and Shenzhen No. 3 People’s Hospital in China. Both datasets contain normal and abnormal chest X-rays with manifestations of TB and include associated radiologist readings.},
	number      = {6},
	urldate     = {2021-07-05},
	journal     = {Quantitative Imaging in Medicine and Surgery},
	author      = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and Wáng, Yì-Xiáng J. and Lu, Pu-Xuan and Thoma, George},
	month       = dec,
	year        = {2014},
	pmid        = {25525580},
	pmcid       = {PMC4256233},
	pages       = {475--477},
}
"""

_SANITIZE_LABELS = {
    "sex": {
        "male": ["M"],
        "female": ["F"],
        "other": ["O"]
    }
}

def _sanitize_label(type_, label):
    type_ = _SANITIZE_LABELS[type_]
    
    for key, value in iteritems(type_):
        if label in value:
            return key

    return label

def sanitize_lines(lines):
    return list(filter(bool, [strip(line) for line in lines]))

def _str_to_int(o):
    stripped = "".join((s for s in o if s.isdigit()))
    stripped = stripped.lstrip("0")
    
    return int(stripped)

def merge_images(*args, **kwargs):
    assert len(args) >= 2

    arr = imageio.imread(args[0])

    for path in args:
        a   = imageio.imread(path)
        arr = np.add(arr, a)

    output = kwargs.get("output")

    if output:
        img = Image.fromarray(arr)
        img.save(output)

    return arr
class Montgomery(GeneratorBasedBuilder):
    """
    Montgomery County Chest X-ray Dataset.
    """

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial Release."
    }

    def _info(self):
        return DatasetInfo(
            builder     = self,
            description = _DATASET_DESCRIPTION,
            features    = FeaturesDict({
                "image": ImageF(encoding_format = "png"),
                 "mask": ImageF(encoding_format = "png"),
                  "sex": ClassLabel(names = iterkeys(_SANITIZE_LABELS["sex"])),
                  "age": Tensor(shape = (), dtype = tf.uint8),
                "label": Text(),
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_extracted = dl_manager.download_and_extract(_DATASET_URL)
        return {
            "data": self._generate_examples(path = osp.join(path_extracted, "MontgomerySet"))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "CXR_png")
        path_data   = osp.join(path, "ClinicalReadings")
        path_masks  = osp.join(path, "ManualMask")
        path_masks_merged = osp.join(path_masks, "merged")

        makedirs(path_masks_merged, exist_ok = True)

        for path_img in glob(osp.join(path_images, "*.png")):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".png")[0]

            path_txt  = osp.join(path_data, "%s.txt" % prefix)

            path_mask = osp.join(path_masks_merged, "%s.png" % prefix)

            if not osp.exists(path_mask):
                path_mask_left  = osp.join(path_masks, "leftMask",  "%s.png" % prefix)
                path_mask_right = osp.join(path_masks, "rightMask", "%s.png" % prefix)

                merge_images(path_mask_left, path_mask_right, output = path_mask)
                
            with open(path_txt) as f:
                content = f.readlines()
                lines   = sanitize_lines(content)

                sex     = _sanitize_label("sex", safe_decode(strip(lines[0].split(": ")[1])))
                age     = _str_to_int(safe_decode(strip(lines[1].split(": ")[1])))
                label   = safe_decode(strip(lines[2]))

                yield prefix, {
                    "image": path_img,
                     "mask": path_mask,
                      "sex": sex,
                      "age": age,
                    "label": label
                }