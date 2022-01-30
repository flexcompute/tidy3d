""" imports interfaces for interacting with server """
import sys

from .webapi import run, upload, get_info, start, monitor, delete, download, load
from .webapi import get_tasks, delete_old
from .container import Job, Batch

from .auth import get_credentials

get_credentials()
