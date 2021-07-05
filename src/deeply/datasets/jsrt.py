import os.path as osp
from glob import glob

from tensorflow_datasets.core import (
    Version,
    GeneratorBasedBuilder,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF
)
import numpy as np
import imageio
from   tqdm import tqdm

from deeply.datasets.montgomery import merge_images
from deeply.util.system import makedirs, read, write
from deeply.log         import get_logger

logger = get_logger()

_DATASET_HOMEPAGE    = "http://db.jsrt.or.jp/eng.php"
_DATASET_DESCRIPTION = """
The standard digital image database with and without chest lung nodules (JSRT database) was created by the Japanese Society of Radiological Technology (JSRT) in cooperation with the Japanese Radiological Society (JRS) in 1998. Since then, the JSRT database has been used by a number of researchers in the world for various research purposes such as image processing, image compression, evaluation of image display, computer-aided diagnosis (CAD), picture archiving and communication system (PACS), and for training and testing.
The number of citations for this database was 35 in 2006, and is likely to increase in the future. After 10 years since the creation of this valuable database, we have decided to release the JSRT database with free of charge in order to facilitate potential users around the world.
"""
_DATASET_CITATION    = """
@article{shiraishi_development_2000,
    title       = {Development of a digital image database for chest radiographs with and without a lung nodule: receiver operating characteristic analysis of radiologists' detection of pulmonary nodules},
    volume      = {174},
    issn        = {0361-803X},
    shorttitle  = {Development of a digital image database for chest radiographs with and without a lung nodule},
    doi         = {10.2214/ajr.174.1.1740071},
    abstract    = {OBJECTIVE: We developed a digital image database (www.macnet.or.jp/jsrt2/cdrom\_nodules.html ) of 247 chest radiographs with and without a lung nodule. The aim of this study was to investigate the characteristics of image databases for potential use in various digital image research projects. Radiologists' detection of solitary pulmonary nodules included in the database was evaluated using a receiver operating characteristic (ROC) analysis.
MATERIALS AND METHODS: One hundred and fifty-four conventional chest radiographs with a lung nodule and 93 radiographs without a nodule were selected from 14 medical centers and were digitized by a laser digitizer with a 2048 x 2048 matrix size (0.175-mm pixels) and a 12-bit gray scale. Lung nodule images were classified into five groups according to the degrees of subtlety shown. The observations of 20 participating radiologists were subjected to ROC analysis for detecting solitary pulmonary nodules. Experimental results (areas under the curve, Az) obtained from observer studies were used for characterization of five groups of lung nodules with different degrees of subtlety.
RESULTS: ROC analysis showed that the database included a wide range of various nodules yielding Az values from 0.574 to 0.991 for the five categories of cases for different degrees of subtlety.
CONCLUSION: This database can be useful for many purposes, including research, education, quality assurance, and other demonstrations.},
    language    = {eng},
    number      = {1},
    journal     = {AJR. American journal of roentgenology},
    author      = {Shiraishi, J. and Katsuragawa, S. and Ikezoe, J. and Matsumoto, T. and Kobayashi, T. and Komatsu, K. and Matsui, M. and Fujita, H. and Kodera, Y. and Doi, K.},
    month       = jan,
    year        = {2000},
    pmid        = {10628457},
    keywords    = {Adult, Aged, Databases as Topic, Female, Humans, Male, Middle Aged, ROC Curve, Radiography, Thoracic, Solitary Pulmonary Nodule},
    pages       = {71--74},
}
"""

_IMAGE_SIZE = (2048, 2048)

class JSRT(GeneratorBasedBuilder):
    """
    JSRT Dataset.
    """

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial Release."
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    You must register and agree to user agreement on the dataset pages:
    http://db.jsrt.or.jp/eng.php and https://www.isi.uu.nl/Research/Databases/SCR respectively.
    Afterwards, you have to put the All247images.zip file and the scr.zip file in the
    manual_dir.
    """

    def _info(self):
        return DatasetInfo(
            builder     = self,
            description = _DATASET_DESCRIPTION,
            features    = FeaturesDict({
                "image": ImageF(),
                 "mask": ImageF()
            }),
            supervised_keys = ("image", "mask"),
            homepage    = _DATASET_HOMEPAGE,
            citation    = _DATASET_CITATION
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.manual_dir
        path_images = osp.join(path, "All247images")
        path_masks  = osp.join(path, "scratch")

        if not osp.exists(path_images) or not osp.exists(path_masks):
            raise ValueError(("You must download the datasets from JSRT website and the ISI website manually",
                " extract and place it into %s" % path))

        # path_extracted_images, path_extracted_masks = dl_manager.extract([path_images, path_masks])
        return {
            "data": self._generate_examples(
                # images_path = osp.join(path_extracted_images, "All247images"),
                # masks_path  = osp.join(path_extracted_masks,  "scratch")
                images_path = path_images,
                masks_path  = path_masks
            )
        }
        
    def _generate_examples(self, images_path, masks_path):
        path_images_png   = osp.join(images_path, "png")
        path_masks_merged = osp.join(masks_path, "merged")

        makedirs(path_images_png,   exist_ok = True)
        makedirs(path_masks_merged, exist_ok = True)

        def get_mask_folder(prefix):
            path_test = osp.join(masks_path, "fold1", "masks", "left lung", "%s.png" % prefix)

            if osp.exists(path_test):
                return "fold1"

            return "fold2"

        logger.debug("Preprocessing images and masks.")

        for path_img in tqdm(glob(osp.join(images_path, "*.IMG"))):
            fname  = osp.basename(osp.normpath(path_img))
            prefix = str(fname).split(".IMG")[0]

            path_img_png = osp.join(path_images_png, "%s.png" % prefix)

            if not osp.exists(path_img_png):
                with open(path_img, mode = "rb") as f:
                    arr = np.fromfile(f, dtype = np.dtype(">u2"))
                    arr = arr.reshape(_IMAGE_SIZE)

                    imageio.imwrite(path_img_png, arr)

            path_mask  = osp.join(path_masks_merged, "%s.png" % prefix)

            if not osp.exists(path_mask):
                folder = get_mask_folder(prefix)

                path_mask_left  = osp.join(masks_path, folder, "masks", "left lung",  "%s.gif" % prefix)
                path_mask_right = osp.join(masks_path, folder, "masks", "right lung", "%s.gif" % prefix)

                if not osp.exists(path_mask_left) or not osp.exists(path_mask_right):
                    logger.warning("Couldn't find lung mask for ID %s" % prefix)
                    continue

                merge_images(path_mask_left, path_mask_right, output = path_mask)

            yield prefix, {
                "image": path_img_png,
                 "mask": path_mask
            }