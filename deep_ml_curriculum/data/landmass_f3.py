from torchvision.datasets import MNIST

class LandmassF3Patches(MNIST):
    """
    LANDMASS is a set of classified 2d subimages extracted from the F3 seismic volume by researchers at Georgia Tech.

    LANDMASS-1, contains 17667 small “patches” of size 99×99 pixels. It includes 9385 Horizon patches, 5140 chaotic patches, 1251 Fault patches, and 1891 Salt Dome patches. The images in this database have values 0-255

    Credits to Agile Geoscience for some of the processing code.

    Source https://dataunderground.org/dataset/landmass-f3
    License: Creative Commons Attribution Share-Alike

    Args:
        root (string): Root directory of dataset where ``LandmassF3/processed/training.pt``
            and  ``LandmassF3/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    classes = ['Chaotic Horizon', 'Fault', 'Horizon', 'Salt Dome']


class LandmassF3PatchesMini(MNIST):
    """
    LANDMASS is a set of classified 2d subimages extracted from the F3 seismic volume by researchers at Georgia Tech.

    LANDMASS-1, contains 17667 small “patches” of size 16x16 pixels. It includes 5500 patches. The images in this database have values 0-255

    Credits to Agile Geoscience for some of the processing code.

    Source https://dataunderground.org/dataset/landmass-f3
    License: Creative Commons Attribution Share-Alike

    Args:
        root (string): Root directory of dataset where ``LandmassF3/processed/training.pt``
            and  ``LandmassF3/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    classes = ['Chaotic Horizon', 'Fault', 'Horizon', 'Salt Dome']
