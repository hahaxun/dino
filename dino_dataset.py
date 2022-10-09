import os
import os.path
import requests
import numpy
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image
from io import BytesIO

from torchvision.datasets.vision import VisionDataset
import gc
import random

session = requests.Session()

def single_json(noteid):
    prefix = "http://nlpfeature.int.xiaohongshu.com/api/feature/str/"
    x = session.get(prefix+noteid)
    return x.json()

class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        #classes, class_to_idx = self.find_classes(self.root)
        samples = open(root,"r").readlines()#self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.id_samples = open(root+".note", "r").readlines()

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
    
    def extract_url_list(self,note_id):
        file_id_prefix = "http://ci.xiaohongshu.com/"
        note = single_json(note_id)
        bbox_mapping = {}
        image_ocr = note.get('image_ocr')
        for _ in image_ocr:
            bbox = []
            detectInfos = _['obj_detect']['detectInfos']
            for detectInfo in detectInfos:
                coordinates = detectInfo['coordinates']
                top_left_x = coordinates[0]['x']
                top_left_y = coordinates[0]['y']
                down_right_x = coordinates[1]['x']
                down_right_y = coordinates[1]['y']
                top_left = (top_left_x,top_left_y)
                down_right = (down_right_x, down_right_y)
                bbox.append((top_left,down_right))
            if len(bbox) > 0:
                bbox_mapping[file_id_prefix + _['file_id']]=bbox
        keys_list=bbox_mapping.keys()
        curr_index = random.randint(0, len(keys_list)) 
        return keys_list[curr_index],bbox_mapping[keys_list[curr_index]][0]

    def run_download(self,url):
        response = session.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        random.seed(index)
        df_margin = random.random()
        if df_margin < 0.3:
            index = random.randint(0, len(self.id_samples) - 1)
            note_id = self.id_samples[index]
            path, bbox = self.extract_url_list(note_id) 

            (up_x, up_y), (bottom_x, bottom_y) = bbox
            target = str(index)
        else:
            path, target = self.samples[index].split(" ")
        
            up_x, up_y, bottom_x, bottom_y, _, _, _, _ = target.split("_")
            up_x, up_y, bottom_x, bottom_y = int(up_x), int(up_y), int(bottom_x), int(bottom_y)

            target = int(path.split("/")[-1].split("_")[0])
        try:
            sample = self.run_download(path)#self.loader(path)
        except Exception as e:
            print(e)
            rgb_array = numpy.random.rand(224,224,3) * 255
            sample = Image.fromarray(rgb_array.astype('uint8')).convert('RGB')
        if self.transform is not None:
            sample_v2 = self.transform(sample,(up_x, up_y, bottom_x, bottom_y))
        if self.target_transform is not None:
            target = self.target_transform(target)
        del sample
        if index % 1000 == 0:
            gc.collect()
        return sample_v2, target

    def __len__(self) -> int:
        return len(self.samples)
