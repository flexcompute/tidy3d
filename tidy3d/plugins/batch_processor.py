
""" 
Point of this plugin:

Provide way for users (and other plugins) to manage multiple job submissions.

way 1: through webAPI

	batch_id = web.submit_batch(simulations=[sim1, sim2, sim3], ...)
	web.monitor_batch(batch_id)
	batch_data = web.load_batch(batch_id)  # td.BatchData object
	for sim_data in batch_data.iter():
		# implicitly download and load one td.SimulationData from batch
		# do stuff ...
		# implicitly delete data / memory for this td.SimulationData

way 2: through batch objects

	batch = Batch(simulations=[sim1, sim2, sim3])
	batch.submit()
	batch.monitor()
	for sim_data in batch_data.load():
		# implicitly download and load one td.SimulationData from batch
		# do stuff ...
		# implicitly delete data / memory for this td.SimulationData		

way 3: leverage existing open source tool. 

	Needs more research

"""

# way 1

BatchId = List[int]

def submit_batch(simulations: List[Simulation]) -> BatchId:
	task_ids = []
	for sim in simulations:
		task_id = web.submit(sim)
		task_ids.append(task_id)
	return task_id

def monitor_batch(batch_id: BatchId) -> None:
	for task_id in batch_id:
		web.monitor(task_id)

def load_batch(batch_id: BatchId) -> None:
	return BatchData(batch_id=batch_id)

class BatchData(pydantic.BaseModel):

	batch_id: BatchId

	def iter(self):
		for task_id in self.batch_id:
			sim_data = web.load(task_id)
			yield sim_data
			# clear sim data

# way 2

class Batch(pydantic.BaseModel):
	simulations: List[Simulation]
	batch_id: Optional[BatchId] = None

	def _assert_submitted(self):
		assert self.batch_id is not None, "batch not subimtted"		

	def submit(self):
		batch_id = []
		for sim in self.simulations:
			task_id = web.submit(sim)
			batch_id.append(task_id)
		self.batch_id = batch_id

	def monitor(self):
		self._assert_submitted()
		for task_id in self.batch_id:
			web.monitor(task_id)

	def load(self):
		self._assert_submitted()
		for task_id in self.batch_id:
			sim_data = web.load(task_id)
			yield sim_data
			# clear sim data
