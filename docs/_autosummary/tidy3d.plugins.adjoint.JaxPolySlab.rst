tidy3d.plugins.adjoint.JaxPolySlab
==================================

.. currentmodule:: tidy3d.plugins.adjoint

.. autoclass:: JaxPolySlab

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~JaxPolySlab.__init__
      ~JaxPolySlab.add_ax_labels_lims
      ~JaxPolySlab.add_type_field
      ~JaxPolySlab.array_to_vertices
      ~JaxPolySlab.bounds_intersection
      ~JaxPolySlab.car_2_sph
      ~JaxPolySlab.car_2_sph_field
      ~JaxPolySlab.compute_dotted_e_d_fields
      ~JaxPolySlab.construct
      ~JaxPolySlab.convert_to_numpy
      ~JaxPolySlab.copy
      ~JaxPolySlab.correct_shape
      ~JaxPolySlab.dict
      ~JaxPolySlab.dict_from_file
      ~JaxPolySlab.dict_from_hdf5
      ~JaxPolySlab.dict_from_hdf5_gz
      ~JaxPolySlab.dict_from_json
      ~JaxPolySlab.dict_from_yaml
      ~JaxPolySlab.edge_contrib
      ~JaxPolySlab.evaluate_inf_shape
      ~JaxPolySlab.from_file
      ~JaxPolySlab.from_gds
      ~JaxPolySlab.from_hdf5
      ~JaxPolySlab.from_hdf5_gz
      ~JaxPolySlab.from_json
      ~JaxPolySlab.from_orm
      ~JaxPolySlab.from_shapely
      ~JaxPolySlab.from_tidy3d
      ~JaxPolySlab.from_yaml
      ~JaxPolySlab.generate_docstring
      ~JaxPolySlab.get_jax_field_names
      ~JaxPolySlab.get_sub_model
      ~JaxPolySlab.get_submodels_by_hash
      ~JaxPolySlab.get_tuple_group_name
      ~JaxPolySlab.get_tuple_index
      ~JaxPolySlab.help
      ~JaxPolySlab.inside
      ~JaxPolySlab.inside_meshgrid
      ~JaxPolySlab.intersections_2dbox
      ~JaxPolySlab.intersections_plane
      ~JaxPolySlab.intersects
      ~JaxPolySlab.intersects_axis_position
      ~JaxPolySlab.intersects_plane
      ~JaxPolySlab.json
      ~JaxPolySlab.kspace_2_sph
      ~JaxPolySlab.limit_number_of_vertices
      ~JaxPolySlab.load_gds_vertices_gdspy
      ~JaxPolySlab.load_gds_vertices_gdstk
      ~JaxPolySlab.make_grad_monitors
      ~JaxPolySlab.no_complex_self_intersecting_polygon_at_reference_plane
      ~JaxPolySlab.no_dilation
      ~JaxPolySlab.no_self_intersecting_polygon_during_extrusion
      ~JaxPolySlab.no_sidewall
      ~JaxPolySlab.parse_file
      ~JaxPolySlab.parse_obj
      ~JaxPolySlab.parse_raw
      ~JaxPolySlab.parse_xyz_kwargs
      ~JaxPolySlab.plot
      ~JaxPolySlab.plot_shape
      ~JaxPolySlab.pop_axis
      ~JaxPolySlab.reflect_points
      ~JaxPolySlab.rotate_points
      ~JaxPolySlab.schema
      ~JaxPolySlab.schema_json
      ~JaxPolySlab.slab_bounds_order
      ~JaxPolySlab.sph_2_car
      ~JaxPolySlab.sph_2_car_field
      ~JaxPolySlab.store_vjp
      ~JaxPolySlab.store_vjp_parallel
      ~JaxPolySlab.store_vjp_sequential
      ~JaxPolySlab.surface_area
      ~JaxPolySlab.to_file
      ~JaxPolySlab.to_hdf5
      ~JaxPolySlab.to_hdf5_gz
      ~JaxPolySlab.to_json
      ~JaxPolySlab.to_list
      ~JaxPolySlab.to_tidy3d
      ~JaxPolySlab.to_yaml
      ~JaxPolySlab.tree_flatten
      ~JaxPolySlab.tree_unflatten
      ~JaxPolySlab.tuple_to_dict
      ~JaxPolySlab.unpop_axis
      ~JaxPolySlab.update_forward_refs
      ~JaxPolySlab.updated_copy
      ~JaxPolySlab.validate
      ~JaxPolySlab.vertex_vjp
      ~JaxPolySlab.vertices_to_array
      ~JaxPolySlab.volume
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~JaxPolySlab.base_polygon
      ~JaxPolySlab.bound_center
      ~JaxPolySlab.bound_size
      ~JaxPolySlab.bounding_box
      ~JaxPolySlab.bounds
      ~JaxPolySlab.center_axis
      ~JaxPolySlab.finite_length_axis
      ~JaxPolySlab.is_ccw
      ~JaxPolySlab.length_axis
      ~JaxPolySlab.middle_polygon
      ~JaxPolySlab.plot_params
      ~JaxPolySlab.reference_polygon
      ~JaxPolySlab.top_polygon
      ~JaxPolySlab.zero_dims
      ~JaxPolySlab.vertices
      ~JaxPolySlab.slab_bounds
      ~JaxPolySlab.dilation
      ~JaxPolySlab.axis
      ~JaxPolySlab.sidewall_angle
      ~JaxPolySlab.reference_plane
   
   