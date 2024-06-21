from .base_datamodule import BaseDataModule
from .datasets.Pet_dataset import PetDataset


class PetDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PetDataset

    @property
    def dataset_name(self):
        return "Pet"


if __name__ == '__main__':
    pass
