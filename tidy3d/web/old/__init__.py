""" import the main callables and try to get credentials from user """
import boto3

from .auth import get_credentials
from .webapi import *
from .job import Job
from .batch import Batch

# should this be set by the config?
boto3.setup_default_session(region_name="us-east-1")

get_credentials()
