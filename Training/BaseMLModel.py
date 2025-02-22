import os
from utils import Logger
import pprint
from datetime import datetime

class BaseMLModel:

	def __init__(self, **params):
		for k, v in params.items():
			setattr(self, k, v)

		now = datetime.now().strftime("%Y%m%d%H%M")
		self.report_directory = \
			os.path.join(".", 'Reports', self.model_name, f"{now}")
		if not os.path.exists(self.report_directory):
			os.makedirs(self.report_directory)

		self.log = Logger(address = f"{self.report_directory}/Log.log")
		self.log.info(pprint.pformat(self.__dict__))