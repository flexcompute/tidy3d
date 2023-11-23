tidy3d.plugins.adjoint.JaxSimulation
====================================

.. currentmodule:: tidy3d.plugins.adjoint

.. autoclass:: JaxSimulation

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~JaxSimulation.__init__
      ~JaxSimulation.add_ax_labels_lims
      ~JaxSimulation.add_type_field
      ~JaxSimulation.bloch_with_symmetry
      ~JaxSimulation.boundaries_for_zero_dims
      ~JaxSimulation.bounds_intersection
      ~JaxSimulation.car_2_sph
      ~JaxSimulation.car_2_sph_field
      ~JaxSimulation.construct
      ~JaxSimulation.copy
      ~JaxSimulation.dict
      ~JaxSimulation.dict_from_file
      ~JaxSimulation.dict_from_hdf5
      ~JaxSimulation.dict_from_hdf5_gz
      ~JaxSimulation.dict_from_json
      ~JaxSimulation.dict_from_yaml
      ~JaxSimulation.diffraction_monitor_boundaries
      ~JaxSimulation.diffraction_monitor_medium
      ~JaxSimulation.discretize
      ~JaxSimulation.discretize_monitor
      ~JaxSimulation.eps_bounds
      ~JaxSimulation.epsilon
      ~JaxSimulation.epsilon_on_grid
      ~JaxSimulation.evaluate_inf_shape
      ~JaxSimulation.from_bounds
      ~JaxSimulation.from_file
      ~JaxSimulation.from_gds
      ~JaxSimulation.from_hdf5
      ~JaxSimulation.from_hdf5_gz
      ~JaxSimulation.from_json
      ~JaxSimulation.from_orm
      ~JaxSimulation.from_shapely
      ~JaxSimulation.from_simulation
      ~JaxSimulation.from_tidy3d
      ~JaxSimulation.from_yaml
      ~JaxSimulation.generate_docstring
      ~JaxSimulation.get_freq_adjoint
      ~JaxSimulation.get_grad_monitors
      ~JaxSimulation.get_jax_field_names
      ~JaxSimulation.get_monitor_by_name
      ~JaxSimulation.get_sub_model
      ~JaxSimulation.get_submodels_by_hash
      ~JaxSimulation.get_tuple_group_name
      ~JaxSimulation.get_tuple_index
      ~JaxSimulation.help
      ~JaxSimulation.inside
      ~JaxSimulation.inside_meshgrid
      ~JaxSimulation.intersecting_media
      ~JaxSimulation.intersecting_structures
      ~JaxSimulation.intersections_2dbox
      ~JaxSimulation.intersections_plane
      ~JaxSimulation.intersections_with
      ~JaxSimulation.intersects
      ~JaxSimulation.intersects_axis_position
      ~JaxSimulation.intersects_plane
      ~JaxSimulation.json
      ~JaxSimulation.kspace_2_sph
      ~JaxSimulation.load_gds_vertices_gdspy
      ~JaxSimulation.load_gds_vertices_gdstk
      ~JaxSimulation.make_sim_fwd
      ~JaxSimulation.monitor_medium
      ~JaxSimulation.parse_file
      ~JaxSimulation.parse_obj
      ~JaxSimulation.parse_raw
      ~JaxSimulation.parse_xyz_kwargs
      ~JaxSimulation.perturbed_mediums_copy
      ~JaxSimulation.plane_wave_boundaries
      ~JaxSimulation.plot
      ~JaxSimulation.plot_3d
      ~JaxSimulation.plot_boundaries
      ~JaxSimulation.plot_eps
      ~JaxSimulation.plot_grid
      ~JaxSimulation.plot_monitors
      ~JaxSimulation.plot_pml
      ~JaxSimulation.plot_shape
      ~JaxSimulation.plot_sources
      ~JaxSimulation.plot_structures
      ~JaxSimulation.plot_structures_eps
      ~JaxSimulation.plot_symmetries
      ~JaxSimulation.pop_axis
      ~JaxSimulation.reflect_points
      ~JaxSimulation.rotate_points
      ~JaxSimulation.schema
      ~JaxSimulation.schema_json
      ~JaxSimulation.sph_2_car
      ~JaxSimulation.sph_2_car_field
      ~JaxSimulation.split_monitors
      ~JaxSimulation.split_structures
      ~JaxSimulation.store_vjp
      ~JaxSimulation.store_vjp_parallel
      ~JaxSimulation.store_vjp_sequential
      ~JaxSimulation.surface_area
      ~JaxSimulation.surfaces
      ~JaxSimulation.surfaces_with_exclusion
      ~JaxSimulation.tfsf_boundaries
      ~JaxSimulation.tfsf_with_symmetry
      ~JaxSimulation.to_file
      ~JaxSimulation.to_hdf5
      ~JaxSimulation.to_hdf5_gz
      ~JaxSimulation.to_json
      ~JaxSimulation.to_simulation
      ~JaxSimulation.to_simulation_fwd
      ~JaxSimulation.to_yaml
      ~JaxSimulation.tree_flatten
      ~JaxSimulation.tree_unflatten
      ~JaxSimulation.tuple_to_dict
      ~JaxSimulation.unpop_axis
      ~JaxSimulation.update_forward_refs
      ~JaxSimulation.updated_copy
      ~JaxSimulation.validate
      ~JaxSimulation.validate_pre_upload
      ~JaxSimulation.volume
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~JaxSimulation.allow_gain
      ~JaxSimulation.background_structure
      ~JaxSimulation.bounding_box
      ~JaxSimulation.bounds
      ~JaxSimulation.bounds_pml
      ~JaxSimulation.complex_fields
      ~JaxSimulation.custom_datasets
      ~JaxSimulation.dt
      ~JaxSimulation.freq_adjoint
      ~JaxSimulation.frequency_range
      ~JaxSimulation.geometry
      ~JaxSimulation.grid
      ~JaxSimulation.medium_map
      ~JaxSimulation.mediums
      ~JaxSimulation.monitors_data_size
      ~JaxSimulation.num_cells
      ~JaxSimulation.num_pml_layers
      ~JaxSimulation.num_time_steps
      ~JaxSimulation.nyquist_step
      ~JaxSimulation.plot_params
      ~JaxSimulation.pml_thicknesses
      ~JaxSimulation.simulation_geometry
      ~JaxSimulation.tmesh
      ~JaxSimulation.volumetric_structures
      ~JaxSimulation.wvl_mat_min
      ~JaxSimulation.zero_dims
      ~JaxSimulation.input_structures
      ~JaxSimulation.output_monitors
      ~JaxSimulation.grad_monitors
      ~JaxSimulation.grad_eps_monitors
      ~JaxSimulation.fwidth_adjoint
      ~JaxSimulation.run_time
      ~JaxSimulation.medium
      ~JaxSimulation.symmetry
      ~JaxSimulation.structures
      ~JaxSimulation.sources
      ~JaxSimulation.boundary_spec
      ~JaxSimulation.monitors
      ~JaxSimulation.grid_spec
      ~JaxSimulation.shutoff
      ~JaxSimulation.subpixel
      ~JaxSimulation.normalize_index
      ~JaxSimulation.courant
      ~JaxSimulation.version
      ~JaxSimulation.size
      ~JaxSimulation.center
   
   