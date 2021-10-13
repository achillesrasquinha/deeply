import os, os.path as osp
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

from bpyutils.util.string  import strip, safe_decode
from bpyutils.util.system  import makedirs
from bpyutils.util.imports import import_or_raise
from bpyutils._compat import iterkeys, iteritems

_DATASET_URL         = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar"
_DATASET_HOMEPAGE    = "https://polyp.grand-challenge.org/CVCClinicDB/"
_DATASET_DESCRIPTION = """
The CVC-ClinicDB database is built in collaboration with Hospital Clinic of Barcelona, Spain. CVC-ClinicDB has been generated from 23 different video studies from standard colonoscopy interventions with white light. CVC-ClinicDB database comprises 612 polyp images of size 576 × 768.
"""
_DATASET_CITATION    = """\
@article{BERNAL201599,
    title    = {WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians},
    journal  = {Computerized Medical Imaging and Graphics},
    volume   = {43},
    pages    = {99-111},
    year     = {2015},
    issn     = {0895-6111},
    doi      = {https://doi.org/10.1016/j.compmedimag.2015.02.007},
    url      = {https://www.sciencedirect.com/science/article/pii/S0895611115000567},
    author   = {Jorge Bernal and F. Javier Sánchez and Gloria Fernández-Esparrach and Debora Gil and Cristina Rodríguez and Fernando Vilariño},
    keywords = {Polyp localization, Energy maps, Colonoscopy, Saliency, Valley detection},
    abstract = {We introduce in this paper a novel polyp localization method for colonoscopy videos. Our method is based on a model of appearance for polyps which defines polyp boundaries in terms of valley information. We propose the integration of valley information in a robust way fostering complete, concave and continuous boundaries typically associated to polyps. This integration is done by using a window of radial sectors which accumulate valley information to create WM-DOVA (Window Median Depth of Valleys Accumulation) energy maps related with the likelihood of polyp presence. We perform a double validation of our maps, which include the introduction of two new databases, including the first, up to our knowledge, fully annotated database with clinical metadata associated. First we assess that the highest value corresponds with the location of the polyp in the image. Second, we show that WM-DOVA energy maps can be comparable with saliency maps obtained from physicians’ fixations obtained via an eye-tracker. Finally, we prove that our method outperforms state-of-the-art computational saliency results. Our method shows good performance, particularly for small polyps which are reported to be the main sources of polyp miss-rate, which indicates the potential applicability of our method in clinical practice.}
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
        arr = np.maximum(arr, a)

    output = kwargs.get("output")

    if output:
        img = Image.fromarray(arr)
        img.save(output)

    return arr

class CVCClinicDB(GeneratorBasedBuilder):
    """
    CVC Clinic DB Dataset.
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
                 "mask": ImageF(encoding_format = "png")
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path_downloaded = dl_manager.download(_DATASET_URL)
        patoolib        = import_or_raise("patoolib", name = "patool")
        print(dir(dl_manager))
        path_extracted  = patoolib.extract_archive(path_downloaded)
        # return {
        #     "data": self._generate_examples(path = osp.join(path_extracted, "CVC-ClinicDB"))
        # }
        
    def _generate_examples(self, path):
        
        pass
        # path_images = osp.join(path, "CXR_png")
        # path_data   = osp.join(path, "ClinicalReadings")
        # path_masks  = osp.join(path, "ManualMask")
        # path_masks_merged = osp.join(path_masks, "merged")

        # makedirs(path_masks_merged, exist_ok = True)

        # for path_img in glob(osp.join(path_images, "*.png")):
        #     fname  = osp.basename(osp.normpath(path_img))
        #     prefix = str(fname).split(".png")[0]

        #     path_txt  = osp.join(path_data, "%s.txt" % prefix)

        #     path_mask = osp.join(path_masks_merged, "%s.png" % prefix)

        #     if not osp.exists(path_mask):
        #         path_mask_left  = osp.join(path_masks, "leftMask",  "%s.png" % prefix)
        #         path_mask_right = osp.join(path_masks, "rightMask", "%s.png" % prefix)

        #         merge_images(path_mask_left, path_mask_right, output = path_mask)
                
        #     with open(path_txt) as f:
        #         content = f.readlines()
        #         lines   = sanitize_lines(content)

        #         sex     = _sanitize_label("sex", safe_decode(strip(lines[0].split(": ")[1])))
        #         age     = _str_to_int(safe_decode(strip(lines[1].split(": ")[1])))
        #         label   = safe_decode(strip(lines[2]))

        #         yield prefix, {
        #             "image": path_img,
        #              "mask": path_mask,
        #               "sex": sex,
        #               "age": age,
        #             "label": label
        #         }