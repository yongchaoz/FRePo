"""
TF: https://github.com/ksachdeva/tiny-imagenet-tfds/blob/master/tiny_imagenet/_imagenet.py
PyTorch: https://gist.github.com/lromor/bcfc69dcf31b2f3244358aea10b7a11b
"""

import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_EXTRACTED_FOLDER_NAME = "tiny-imagenet-200"

SUPPORTED_IMAGE_FORMAT = (".jpg", ".jpeg", ".png")


def _list_folders(root_dir):
    return [
        f for f in tf.io.gfile.listdir(root_dir)
        if tf.io.gfile.isdir(os.path.join(root_dir, f))
    ]


def _list_imgs(root_dir):
    return [
        os.path.join(root_dir, f)
        for f in tf.io.gfile.listdir(root_dir)
        if any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMAT)
    ]


class_names = ['Egyptian cat', 'reel', 'volleyball', 'rocking chair', 'lemon', 'bullfrog', 'basketball', 'cliff',
               'espresso', 'plunger', 'parking meter', 'German shepherd', 'dining table', 'monarch', 'brown bear',
               'school bus', 'pizza', 'guinea pig', 'umbrella', 'organ', 'oboe', 'maypole', 'goldfish', 'potpie',
               'hourglass', 'seashore', 'computer keyboard', 'Arabian camel', 'ice cream', 'nail', 'space heater',
               'cardigan', 'baboon', 'snail', 'coral reef', 'albatross', 'spider web', 'sea cucumber', 'backpack',
               'Labrador retriever', 'pretzel', 'king penguin', 'sulphur butterfly', 'tarantula', 'lesser panda',
               'pop bottle', 'banana', 'sock', 'cockroach', 'projectile', 'beer bottle', 'mantis', 'freight car',
               'guacamole', 'remote control', 'European fire salamander', 'lakeside', 'chimpanzee', 'pay-phone',
               'fur coat', 'alp', 'lampshade', 'torch', 'abacus', 'moving van', 'barrel', 'tabby', 'goose', 'koala',
               'bullet train', 'CD player', 'teapot', 'birdhouse', 'gazelle', 'academic gown', 'tractor', 'ladybug',
               'miniskirt', 'golden retriever', 'triumphal arch', 'cannon', 'neck brace', 'sombrero', 'gasmask',
               'candle', 'desk', 'frying pan', 'bee', 'dam', 'spiny lobster', 'police van', 'iPod', 'punching bag',
               'beacon', 'jellyfish', 'wok', "potter's wheel", 'sandal', 'pill bottle', 'butcher shop', 'slug', 'hog',
               'cougar', 'crane', 'vestment', 'dragonfly', 'cash machine', 'mushroom', 'jinrikisha', 'water tower',
               'chest', 'snorkel', 'sunglasses', 'fly', 'limousine', 'black stork', 'dugong', 'sports car', 'water jug',
               'suspension bridge', 'ox', 'ice lolly', 'turnstile', 'Christmas stocking', 'broom', 'scorpion',
               'wooden spoon', 'picket fence', 'rugby ball', 'sewing machine', 'steel arch bridge', 'Persian cat',
               'refrigerator', 'barn', 'apron', 'Yorkshire terrier', 'swimming trunks', 'stopwatch', 'lawn mower',
               'thatch', 'fountain', 'black widow', 'bikini', 'plate', 'teddy', 'barbershop', 'confectionery',
               'beach wagon', 'scoreboard', 'orange', 'flagpole', 'American lobster', 'trolleybus', 'drumstick',
               'dumbbell', 'brass', 'bow tie', 'convertible', 'bighorn', 'orangutan', 'American alligator', 'centipede',
               'syringe', 'go-kart', 'brain coral', 'sea slug', 'cliff dwelling', 'mashed potato', 'viaduct',
               'military uniform', 'pomegranate', 'chain', 'kimono', 'comic book', 'trilobite', 'bison', 'pole',
               'boa constrictor', 'poncho', 'bathtub', 'grasshopper', 'walking stick', 'Chihuahua', 'tailed frog',
               'lion', 'altar', 'obelisk', 'beaker', 'bell pepper', 'bannister', 'bucket', 'magnetic compass',
               'meat loaf', 'gondola', 'standard poodle', 'acorn', 'lifeboat', 'binoculars', 'cauliflower',
               'African elephant']

# Use V2 to avoid name collision with tfds
class TinyImagenetV2(tfds.core.GeneratorBasedBuilder):
    """ tiny-imagenet dataset """
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("""Tiny ImageNet Challenge is a similar challenge as ImageNet with a smaller dataset but
                         less image classes. It contains 200 image classes, a training
                         dataset of 100, 000 images, a validation dataset of 10, 000
                         images, and a test dataset of 10, 000 images. All images are
                         of size 64×64."""),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(64, 64, 3), encoding_format="jpeg"),
                "id": tfds.features.Text(),
                "label": tfds.features.ClassLabel(names=class_names),
            }),
            supervised_keys=("image", "label"),
            homepage="https://tiny-imagenet.herokuapp.com/",
            citation=r"""@article{tiny-imagenet,
                              author = {Li,Fei-Fei}, {Karpathy,Andrej} and {Johnson,Justin}"}""",
        )

    def _process_train_ds(self, ds_folder, identities):
        path_to_ds = os.path.join(ds_folder, 'train')
        names = _list_folders(path_to_ds)

        label_images = {}
        for n in names:
            images_dir = os.path.join(path_to_ds, n, 'images')
            total_images = _list_imgs(images_dir)
            label_images[n] = {
                'images': total_images,
                'id': identities.index(n)
            }

        return label_images

    def _process_test_ds(self, ds_folder, identities):
        path_to_ds = os.path.join(ds_folder, 'val')

        # read the val_annotations.txt file
        with tf.io.gfile.GFile(os.path.join(path_to_ds, 'val_annotations.txt')) as f:
            data_raw = f.read()

        lines = data_raw.split("\n")

        label_images = {}
        for line in lines:
            if line == '':
                continue
            row_values = line.strip().split()
            label_name = row_values[1]
            if not label_name in label_images.keys():
                label_images[label_name] = {
                    'images': [],
                    'id': identities.index(label_name)
                }

            label_images[label_name]['images'].append(
                os.path.join(path_to_ds, 'images', row_values[0]))

        return label_images

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.extract(dl_manager.download(_URL))

        ds_folder = os.path.join(extracted_path, _EXTRACTED_FOLDER_NAME)

        with tf.io.gfile.GFile(os.path.join(ds_folder, 'wnids.txt')) as f:
            data_raw = f.read()

        lines = data_raw.split("\n")

        train_label_images = self._process_train_ds(ds_folder, lines)
        test_label_images = self._process_test_ds(ds_folder, lines)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(label_images=train_label_images, )),

            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(label_images=test_label_images, )),
        ]

    def _generate_examples(self, label_images):
        for label, image_info in label_images.items():
            for image_path in image_info['images']:
                key = "%s/%s" % (label, os.path.basename(image_path))
                yield key, {
                    "image": image_path,
                    "id": label,
                    "label": image_info['id'],
                }
