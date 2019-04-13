import os
import numpy as np

from torch.utils.data import Dataset
from nlp.data.utils import download_url, makedir_exist_ok


class Anon(Dataset):
    """Anonymous Synthetic Dataset.

    Parameters
    ----------
    root str :
        Root directory of dataset where ``processed/training.npy``
        ``processed/validation.npy and ``processed/test.npy`` exist.

    partition : str
        dataset partition to be loaded.
        Either 'train', 'validation', or 'test'.

    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    """
    urls = [
      'https://raw.githubusercontent.com/yngtodd/unlp/master/anon/train-data.npy',
      'https://raw.githubusercontent.com/yngtodd/unlp/master/anon/train-labels.npy',
      'https://raw.githubusercontent.com/yngtodd/unlp/master/anon/test-data.npy',
      'https://raw.githubusercontent.com/yngtodd/unlp/master/anon/test-labels.npy'
    ]

    training_data_file= 'train_data.npy'
    training_label_file = 'train_labels.npy'
    test_data_file = 'test_data.npy'
    test_label_file = 'test_labels.npy'

    def __init__(self, root, partition, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.partition = partition
        if self.partition == 'train':
            data_file = self.training_data_file
            label_file = self.training_label_file
        elif self.partition == 'test':
            data_file = self.test_data_file
            label_file = self.test_label_file
        else:
            raise ValueError("Partition must either be 'train' or 'test'.")

        self.data = np.load(os.path.join(self.processed_folder, data_file))
        self.targets = np.load(os.path.join(self.processed_folder, label_file))

    def __len__(self):
        return len(self.data)

    def load_data(self):
        return self.data, self.targets

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        index : int
          Index of the data to be loaded.

        Returns
        -------
        (document, target) : tuple
           where target is index of the target class.
        """
        document, target = self.data[idx], int(self.targets[idx])


        if self.transform is not None:
            document = self.transform(document)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return document, target

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_data_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.training_label_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_data_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_label_file))

    @staticmethod
    def extract_array(path, remove_finished=False):
        print('Extracting {}'.format(path))
        arry = np.load(path)
        if remove_finished:
            os.unlink(path)

    def download(self):
        """Download the Synthetic data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_array(path=file_path, remove_finished=False)

        # process and save as numpy files
        print('Processing...')

        training_set = (
            np.load(os.path.join(self.raw_folder, 'train-data.npy')),
            np.load(os.path.join(self.raw_folder, 'train-labels.npy'))
        )
        test_set = (
            np.load(os.path.join(self.raw_folder, 'test-data.npy')),
            np.load(os.path.join(self.raw_folder, 'test-labels.npy'))
        )

        # Save processed training data
        train_data_path = os.path.join(self.processed_folder, self.training_data_file)
        np.save(train_data_path, training_set[0])
        train_label_path = os.path.join(self.processed_folder, self.training_label_file)
        np.save(train_label_path, training_set[1])

        #Save processed test data
        test_data_path = os.path.join(self.processed_folder, self.test_data_file)
        np.save(test_data_path, test_set[0])
        test_label_path = os.path.join(self.processed_folder, self.test_label_file)
        np.save(test_label_path, test_set[1])

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.partition
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
