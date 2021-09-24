""" command-line interface for instructions run `python tidy3d.py --help` """

import argparse
import pyfiglet
from pprint import pprint

from tidy3d import Simulation
from tidy3d.web import Job

ascii_banner = pyfiglet.figlet_format("Tidy3D")
print(ascii_banner)

parser = argparse.ArgumentParser(description=ascii_banner)

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
    simulation = Simulation.load_yaml(sim_file)
else:
    simulation = Simulation.load(sim_file)

# inspect the simulation
if inspect_sim:
    ascii_banner = pyfiglet.figlet_format("Simulation Plot")
    print(ascii_banner)
    looks_good = input("Do you want to continue to submit? [y]/[n]")
    if looks_good.lower() != "y":
        print(" - exiting")
        exit(0)

# upload to server
job = Job(simulation=simulation, task_name=task_name)
job.upload()

# inspect credit and data usage
if inspect_credits:
    info = job.get_info()
    print(
        f'task "{task_name}" estimated to use '
        f"\n\t{info.credits:.2f} credits and "
        f"\n\t{info.size_bytes:.2e} bytes of storage."
    )
    looks_good = input("Do you want to continue to submit? [y]/[n]")
    if looks_good.lower() != "y":
        print(" - exiting")
        exit(0)

# run the simulation and load results
job.run()
job.monitor()
sim_data = job.load_results(path=out_file)

# visualize results
if viz_results:
    ascii_banner = pyfiglet.figlet_format("Simulation Data")
    print(ascii_banner)
