""" handles filesystem, storage """
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


def get_s3_user():
    """gets all information relevant to user's Config for s3"""
    client = get_s3_client()
    bucket = Config.studio_bucket
    user_id = Config.user["UserId"]
    return client, bucket, user_id


class UploadProgress:
    """updates progressbar for the upload status"""

    def __init__(self, size_bytes, progress):
        """initialize with the size of file and rich.progress.Progress() instance"""
        self.progress = progress
        self.ul_task = self.progress.add_task("[red]Uploading...", total=size_bytes)

    def report(self, bytes_in_chunk):
        """the progressbar with recent chunk"""
        self.progress.update(self.ul_task, advance=bytes_in_chunk)


class DownloadProgress:
    """updates progressbar for the download status"""

    def __init__(self, size_bytes, progress):
        """initialize with the size of file and rich.progress.Progress() instance"""
        self.progress = progress
        self.dl_task = self.progress.add_task("[red]Downloading...", total=size_bytes)

    def report(self, bytes_in_chunk):
        """the progressbar with recent chunk"""
        self.progress.update(self.dl_task, advance=bytes_in_chunk)
