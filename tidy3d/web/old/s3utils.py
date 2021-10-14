""" handles filesystem, storage """
import sys

import boto3

from .config import DEFAULT_CONFIG as Config


def get_s3_client():
    """returns the client based on Config"""
    keys = Config.user
    return boto3.client(
        "s3",
        aws_access_key_id=keys["userAccessKey"],
        aws_secret_access_key=keys["userSecretAccessKey"],
        region_name=Config.s3_region,
    )


class UploadProgress:
    """stores and prints the upload status"""

    def __init__(self, size):
        """initialize with size of file"""
        self.size = size
        self.uploaded_so_far = 0.0

    def report(self, bytes_in_chunk):
        """update and report status"""
        self.uploaded_so_far += bytes_in_chunk
        perc_done = 100.0 * float(self.uploaded_so_far) / self.size
        message = f"file upload progress: {perc_done:2.2f} %"

        # do we relly want to print to stdout? how bout call this to get %?
        sys.stdout.write(message)
        sys.stdout.flush()


class DownloadProgress:
    """stores and print the download status"""

    def __init__(self, client, bucket, key):
        """initialize with size of file"""
        head_object = client.head_object(Bucket=bucket, Key=key)
        self.size = head_object["ContentLength"]
        self.downloaded_so_far = 0.0

    def report(self, bytes_in_chunk):
        """update and report status"""
        self.downloaded_so_far += bytes_in_chunk
        perc_done = 100.0 * float(self.downloaded_so_far) / self.size

        # do we relly want to print to stdout? how bout call this to get %?
        message = f"file download progress: {perc_done:2.2f} %"
        sys.stdout.write(message)
        sys.stdout.flush()
