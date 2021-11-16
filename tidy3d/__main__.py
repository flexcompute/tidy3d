""" command-line interface. For instructions run `python -m tidy3d --help` """
import sys
import argparse

from tidy3d import Simulation
from tidy3d.web import Job

parser = argparse.ArgumentParser(description="Tidy3D")

parser.add_argument("simulation", help="path to the .json or .yaml file containing the simulation")

parser.add_argument(
    "--out", "-o", default="simulation.hdf5", required=False, help="path to output the data"
)

parser.add_argument(
    "--inspect_sim",
    "-i",
    required=False,
    action="store_true",
    help="visualize simulation and prompt before submitting",
)

parser.add_argument(
    "--inspect_credits",
    "-c",
    required=False,
    action="store_true",
    help="visualize simulation and prompt before submitting",
)

parser.add_argument(
    "--task_name", "-t", default="my_task", required=False, help="set name for task"
)

parser.add_argument(
    "--viz_results",
    "-v",
    action="store_true",
    required=False,
    help="visualize results after submitting",
)

args = parser.parse_args()

sim_file = args.simulation
out_file = args.out
inspect_sim = args.inspect_sim
inspect_credits = args.inspect_credits
task_name = args.task_name
viz_results = args.viz_results

print("simulation file: ", sim_file)
print("data output file: ", out_file)
print("inspect simulation: ", inspect_sim)
print("inspect credits: ", inspect_credits)
print("task name: ", task_name)
print("visualize results: ", viz_results)

""" main script """

# load the simulation
if ".yaml" in sim_file or ".yml" in sim_file:
    simulation = Simulation.from_yaml(sim_file)
else:
    simulation = Simulation.from_file(sim_file)

# inspect the simulation
if inspect_sim:
    looks_good = input("Do you want to continue to submit? [y]/[n]")
    if looks_good.lower() != "y":
        print(" - exiting")
        sys.exit()

# upload to server
job = Job(simulation=simulation, task_name=task_name)
job.upload()

# inspect credit and data usage
if inspect_credits:
    info = job.get_info()
    print(
        f'task "{task_name}" estimated to use '
        f"\n\t{info.optSolverUnit:.2f} credits and "
        f"\n\t{info.s3Storage:.2e} bytes of storage."
    )
    looks_good = input("Do you want to continue to submit? [y]/[n]")
    if looks_good.lower() != "y":
        print(" - exiting")
        sys.exit()

# run the simulation and load results
job.start()
job.monitor()
sim_data = job.load(path=out_file)

# visualize results
if viz_results:
    pass
