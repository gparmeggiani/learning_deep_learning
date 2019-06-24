#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giacomo Parmeggiani <giacomo.parmeggiani@gmail.com>

Download and extract datasets from the internet.

Usage:
python get.py [dataset]

Currently supported datasets:
cifar10     The CIFAR 10 dataset
cifar100    The CIFAR 100 dataset

"""
import sys
import os
import hashlib
from urllib.request import urlretrieve
import tarfile


datasets_urls = {
    'cifar10': ('https://www.cs.toronto.edu/~kriz/', 'cifar-10-python.tar.gz', "c58f30108f718f92721af3b95e74349a"),
    'cifar100': ('https://www.cs.toronto.edu/~kriz/', 'cifar-100-python.tar.gz', "eb9058c3a382ffc7106e4002c42a8d85")
}


def download(base_url, file_name, md5_hash):
    """
    Download a file and check its MD5 checksum.
    If the file already exists and it is not corrupted, the download is skipped

    :param base_url: Base url of the file. Trailing slash needed
    :param file_name: The filename
    :param md5_hash: The MD5 hash of the file
    :return: True if the operation completed successfully
    """

    # Check if the file is already available
    if os.path.isfile(file_name):
        if check_md5(file_name, md5_hash):
            print("\'{}\' exists already with the correct MD5 hash".format(file_name))
            return True
        else:
            print("\'{}\' exists already, but it has a bad MD5 hash. Downloading a new copy of the file".format(file_name))

    print("Downloading {}".format(file_name))

    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:  # near the end
                sys.stderr.write("\n")
        else:  # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

    urlretrieve(base_url+file_name, file_name, reporthook)

    if check_md5(file_name, md5_hash):
        print("Download completed.")
        return True
    else:
        print("Download failed: MD5 mismatch")
        return False


def md5(file_name):
    """
    Calculate the MD5 hash of a file
    
    :param file_name: the neame of the file
    :return: 
    """

    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_md5(file_name, real_hash):
    """

    :param file_name:
    :param real_hash:
    :return:
    """
    file_hash = md5(file_name)
    return file_hash == real_hash


def tar_extract(file_name):
    """
    Extract the .tar.gz file
    :param file_name:
    :return:
    """
    tar = tarfile.open(file_name)
    sys.stdout.flush()
    tar.extractall("./")
    tar.close()


def print_usage():
    """
    Print the usage of this script

    :return:
    """
    print("""Usage: python get.py [dataset]

[dataset]:
cifar10     CIFAR 10
cifar100    CIFAR 100""")


def main():
    """
    Main function
    :return:
    """

    # Make sure we are using Python 3
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    # Print usage if the argument is missing
    dataset_name = None
    try:
        dataset_name = str(sys.argv[1])
    except IndexError:
        print_usage()
        exit(2)

    # Download the dataset
    file_name = ""
    try:
        base_url, file_name, md5_hash = datasets_urls[dataset_name.lower()]
        download(base_url, file_name, md5_hash)

    except KeyError:
        print("Unknown dataset \'dataset_name\'")
        exit(2)

    if file_name.endswith(".tar.gz"):
        print("Extracting files")
        tar_extract(file_name)


if __name__ == '__main__':
    main()
