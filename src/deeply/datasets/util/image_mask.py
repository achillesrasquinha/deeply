import os, os.path as osp

from bpyutils.util.system import get_files

from tensorflow_datasets.core import (
    Version,
    DatasetInfo
)
from tensorflow_datasets.core.features import (
    FeaturesDict,
    Image as ImageF
)

def _info(builder, description = None, homepage = None, citation = None):
    return DatasetInfo(
        builder     = builder,
        description = description,
        features    = FeaturesDict({
            "image": ImageF(encoding_format = "jpeg"),
             "mask": ImageF(encoding_format = "jpeg")
        }),
        supervised_keys = ("image", "mask"),
        homepage    = homepage,
        citation    = citation
    )

def _split_generators(builder, dl_manager, kaggle):
    path_extracted  = dl_manager.download_kaggle_data(kaggle)
    folder_name     = os.listdir(path_extracted)[0]

    return {
        "data": builder._generate_examples(path = osp.join(path_extracted, folder_name))
    }
 
def _generate_examples(builder, path):
    path_images = osp.join(path, "images")
    path_masks  = osp.join(path, "masks")

    for path_img in get_files(path_images):
        fname     = osp.basename(osp.normpath(path_img))
        prefix, _ = osp.splitext(fname)

        path_mask = osp.join(path_masks, fname)

        yield prefix, {
            "image": path_img,
             "mask": path_mask
        }