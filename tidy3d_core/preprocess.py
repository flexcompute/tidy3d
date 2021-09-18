from schema_ops import load_json, validate_schema

""" Loads the JSON file into Simulation and prepares data for solver """

def load_sim_dict(fname_sim: str) -> dict:
	sim_dict = load_json(fname_sim)
	validate_schema(sim_dict)
	return sim_dict
