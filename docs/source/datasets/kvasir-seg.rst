kvasir-seg
==========

.. image:: https://img.shields.io/badge/paperswithcode-📝-00cec9.svg?style=flat-square
    :target: https://paperswithcode.com/sota/medical-imgseg-on-cvc-clinicdb

.. image:: https://img.shields.io/badge/tasks-imgseg-8e44ad.svg?style=flat-square

usage
-----

   >>> import deeply.datasets as dd
   >>> dataset = dd.load("kvasir_seg")

summary
-------

Pixel-wise image segmentation is a highly demanding task in medical image analysis. It is difficult to find annotated medical images with corresponding segmentation mask. Here, we present Kvasir-SEG. It is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks, manually annotated and verified by an experienced gastroenterologist. This work will be valuable for researchers to reproduce results and compare their methods in the future. By adding segmentation masks to the Kvasir dataset, which until today only consisted of framewise annotations, we enable multimedia and computer vision researchers to contribute in the field of polyp segmentation and automatic analysis of colonoscopy videos.

citation
--------

Bernal, Jorge, et al. “WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians.” Computerized Medical Imaging and Graphics, vol. 43, July 2015, pp. 99–111. ScienceDirect, https://doi.org/10.1016/j.compmedimag.2015.02.007.
---. “WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians.” Computerized Medical Imaging and Graphics: The Official Journal of the Computerized Medical Imaging Society, vol. 43, July 2015, pp. 99–111. PubMed, https://doi.org/10.1016/j.compmedimag.2015.02.007.

contributors
------------

Thanks to `@achillesrasquinha <https://github.com/achillesrasquinha>`_ for adding this dataset.