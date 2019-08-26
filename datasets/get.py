#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and extract datasets from the internet.

Usage:
python get.py [dataset]

Currently supported datasets:
cifar10     The CIFAR 10 dataset
cifar100    The CIFAR 100 dataset
iris        Iris flower data set

"""
import sys
import os
import hashlib
from urllib.request import urlretrieve
import tarfile

__author__ = "Giacomo Parmeggiani <giacomo.parmeggiani@gmail.com>"

datasets_urls = {
    'cifar10': [('https://www.cs.toronto.edu/~kriz/', 'cifar-10-python.tar.gz', "c58f30108f718f92721af3b95e74349a")],
    'cifar100': [('https://www.cs.toronto.edu/~kriz/', 'cifar-100-python.tar.gz', "eb9058c3a382ffc7106e4002c42a8d85")],
    'iris': [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/', 'iris.data', "42615765a885ddf54427f12c34a0a070"),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/', 'iris.names', "1759b220ad1e305d573a7dbbdb9c7039"),
    ]
}


def download(base_url, file_name, md5_hash=None):
    """Download a file and check its MD5 checksum, if provided.

    If the file already exists and it is not corrupted, the download is skipped

    Args:
        (str) base_url: Base url of the file. Trailing slash needed
        (str) file_name: The filename
        (str) md5_hash: The MD5 hash of the file
    
    Returns: 
        True if the operation completed successfully
    """

    # Check if the file is already available
    if md5_hash is not None and os.path.isfile(file_name):
        if check_file_md5(file_name, md5_hash):
            print("'{}' already exists with the correct MD5 hash".format(file_name))
            return True
        else:
            print("'{}' already exists, however it has a bad MD5 hash. Downloading a new copy of the file".format(file_name))

    print("Downloading '{}'".format(file_name))

    def reporthook(blocknum, blocksize, totalsize):
        """Helper function used to print the download progress
        """
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = min(100, readsofar * 100 / totalsize)
            s = "\r%5.1f%% %*d / %d" % (percent, len(str(totalsize)), min(readsofar, totalsize), totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:
                sys.stderr.write("\n")
        else:
            sys.stderr.write("read %d\n" % (readsofar,))

    urlretrieve(
        base_url+file_name, 
        filename=file_name, 
        reporthook=reporthook)


    if md5_hash is not None and not check_file_md5(file_name, md5_hash):
        print("Download failed: MD5 mismatch")
        return False

    else:
        print("Download completed.")
        return True


def file_md5(file_name):
    """Calculate the MD5 hash of a file.
    
    Args:
        (str) file_name: the name of the file
    
    Returns:
        (str) The hex string representation of the MD5 digest of the file
    """

    hash_md5 = hashlib.md5()

    with open(file_name, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def check_file_md5(file_name, expected_hash):
    """ Check if a file has the expected MD5 hash

    Args:
        (str) file_name: The name of the file you want to check the MD5 hash of
        (str) expected_hash: The expected hash for the file
    
    Returns:
        (boolean) True if the file's hash matches the expected MD5 hash
    """
    return file_md5(file_name) == expected_hash


def tar_extract(file_name):
    """ Extract a .tar.gz file

    Args:
        (str) file_name: The name of the .tar.gz file to extract
    """
    tar = tarfile.open(file_name)
    sys.stdout.flush()
    tar.extractall("./")
    tar.close()


def print_usage():
    """Print the usage of this script.
    """
    print("""Usage: python get.py [dataset]

[dataset]:
cifar10     CIFAR 10 dataset
cifar100    CIFAR 100 dataset
iris        Iris flower data set""")


def main():
    """Main function.
    
    Parse the command line arguments and download the necessary files for the dataset
    """

    # Make sure we are using Python 3
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    # Print usage if the argument is missing
    dataset_name = None
    try:
        dataset_name = str(sys.argv[1]).lower()
    except IndexError:
        print_usage()
        exit(2)

    # Get the datasets info
    try:    
        dataset = datasets_urls[dataset_name]
    except KeyError:
        print("Unknown dataset '{}'".format(dataset_name))
        exit(2)

    # create a folder if the dataset contains more than one file
    if len(dataset) > 1:
        if not os.path.exists(dataset_name):
            os.makedirs(dataset_name)
            print("Creating '{}' folder".format(dataset_name))
            os.chdir(dataset_name)

    # Download the dataset file(s)
    for base_url, file_name, md5_hash in dataset:
        download(base_url, file_name, md5_hash)

        if file_name.endswith(".tar.gz"):
            print("Extracting {}".format(file_name))
            tar_extract(file_name)


if __name__ == '__main__':
    main()
