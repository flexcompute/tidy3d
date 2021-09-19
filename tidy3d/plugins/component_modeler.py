""" 
Point of this plugin:

Provide way for users (and other plugins) to model their structures as components in a larger model, for example a photonic integrated circuit model.

For now, we can consider this mainly an S-matrix calculation tool, but we can add functionality to it.

"""

PORT_SOURCE_NAME = 'port_source'

class ComponentModel(pydantic.BaseModel):

	base_simulation: Simulation
	source_time: SourceTime
	source_mode: Mode
	port_monitors: Dict[str, ModeMonitor]

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.batch = self.assemble_simulations()

	def assemble_simulations(self):
		""" makes a simulation with source at each port in port monitors """

		simulations = []

		# add all of the port monitors to the base simulation, make temporary copy
		sim_tmp = self.base_simulation.copy()
		sim_tmp[name] = monitor for (name, monitor) in self.port_monitors.items()

		for name, monitor in self.port_monitors.items():

			# add a source just 'outside' this monitor
			port_source = ModeSource(
				center=monitor.center,
				size=monitor.size,		
				source_time=self.source_time,
				mode=source_mode,
			)

			# make a temporary simulation for this port and add this source to it
			sim_tmp_port = self.base_simulation.copy()
			sim_tmp_port.sources[PORT_SOURCE_NAME] = port_source

			# add this simulation to list
			sims.append(sim_tmp_port)

		return Batch(simulations=simulations)

	def submit(self):
		self.batch.submit()

	def monitor(self):
		self.batch.monitor()

	def load(self):
		self.batch.load()

	def compute_S_matrix(self):
		""" compute S-matrix from the batch results (to-do) """
		pass

	def visualize(self):
		""" visualize S-matrix results """
		pass




