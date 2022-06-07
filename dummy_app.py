from tidy3d.plugins import SimulationDataApp

data_fname = "./data/viz_data.hdf5"

app = SimulationDataApp.from_file(data_fname, mode="python")
app.run(debug=True)

# from tidy3d import check_client_version
#
# check_client_version()
