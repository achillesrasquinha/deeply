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
    Image as ImageF
)
from PIL import Image

from bpyutils.util.imports import import_or_raise

_DATASET_URL         = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"
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

def _tiff_to_jpeg(source, dest):
    img = Image.open(source)
    img.save(dest, "JPEG", quality = 100)

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
        path_target     = osp.join(dl_manager.download_dir, "CVC-ClinicDB")
        
        if not osp.exists(path_target):
            patoolib       = import_or_raise("patoolib", name = "patool")
            patoolib.extract_archive(path_downloaded, outdir = dl_manager.download_dir)

        return {
            "data": self._generate_examples(path = osp.join(path_target, ""))
        }
        
    def _generate_examples(self, path):
        path_images = osp.join(path, "Original")
        path_masks  = osp.join(path, "Ground Truth")

        for path_img_tif in glob(osp.join(path_images, "*.tif")):
            fname     = osp.basename(osp.normpath(path_img_tif))
            prefix    = str(fname).split(".tif")[0]

            path_img  = osp.join(path_images, "%s.jpg" % prefix)
            if not osp.exists(path_img):
                _tiff_to_jpeg(path_img_tif, path_img)

            path_mask_tif = osp.join(path_masks, fname)
            path_mask     = osp.join(path_masks, "%s.jpg" % prefix)

            if not osp.exists(path_mask):
                _tiff_to_jpeg(path_mask_tif, path_mask)

            yield fname, {
                "image": path_img,
                 "mask": path_mask
            }