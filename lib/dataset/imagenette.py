"""Imagenette: a subset of 10 easily classified classes from Imagenet.
(tench, English springer, cassette player, chain saw, church, French horn,
garbage truck, gas pump, golf ball, parachute)

"""

import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

_CITATION = """
@misc{imagenette,
  author    = "Jeremy Howard",
  title     = "imagenette",
  url       = "https://github.com/fastai/imagenette/"
}
"""

_DESCRIPTION = """\
Imagenette is a subset of 10 easily classified classes from the Imagenet
dataset. It was originally prepared by Jeremy Howard of FastAI. The objective
behind putting together a small version of the Imagenet dataset was mainly
because running new ideas/algorithms/experiments on the whole Imagenet take a
lot of time.
Note: The v2 config correspond to the new 70/30 train/valid split (released
in Dec 6 2019).
"""

lbl_dict = {
    'n01440764': 'tench',
    'n02102040': 'english springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'french horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}

# Use V2 to avoid name collision with tfds
class ImagenetteV2(tfds.core.GeneratorBasedBuilder):
    """A smaller subset of 10 easily classified classes from Imagenet."""
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(
                    names=['tench', 'english springer', 'cassette player', 'chain saw', 'church', 'french horn',
                           'garbage truck', 'gas pump', 'golf ball', 'parachute']),
            }),
            supervised_keys=("image", "label"),
            homepage="https://github.com/fastai/imagenette",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Generate Splits"""
        extracted_path = dl_manager.download_and_extract(_IMAGENETTE_URL)
        extracted_path = os.path.join(extracted_path, 'imagenette2-160')
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "train"),
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "val"),
                }),
        ]

    def _generate_examples(self, images_dir_path):
        """Generate examples given the image directory path"""
        for image_folder in tf.io.gfile.listdir(images_dir_path):
            for image_file in tf.io.gfile.listdir(os.path.join(images_dir_path,
                                                               image_folder)):
                yield image_file, {
                    'image': '{}/{}/{}'.format(images_dir_path, image_folder,
                                               image_file),
                    'label': lbl_dict[image_folder]
                }
