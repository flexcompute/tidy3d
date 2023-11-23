tidy3d.plugins.adjoint.JaxCustomMedium
======================================

.. currentmodule:: tidy3d.plugins.adjoint

.. autoclass:: JaxCustomMedium

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~JaxCustomMedium.__init__
      ~JaxCustomMedium.add_type_field
      ~JaxCustomMedium.construct
      ~JaxCustomMedium.copy
      ~JaxCustomMedium.d_eps_map
      ~JaxCustomMedium.dict
      ~JaxCustomMedium.dict_from_file
      ~JaxCustomMedium.dict_from_hdf5
      ~JaxCustomMedium.dict_from_hdf5_gz
      ~JaxCustomMedium.dict_from_json
      ~JaxCustomMedium.dict_from_yaml
      ~JaxCustomMedium.e_mult_volume
      ~JaxCustomMedium.eps_comp
      ~JaxCustomMedium.eps_comp_on_grid
      ~JaxCustomMedium.eps_complex_to_eps_sigma
      ~JaxCustomMedium.eps_complex_to_nk
      ~JaxCustomMedium.eps_dataarray_freq
      ~JaxCustomMedium.eps_diagonal
      ~JaxCustomMedium.eps_diagonal_on_grid
      ~JaxCustomMedium.eps_model
      ~JaxCustomMedium.eps_sigma_to_eps_complex
      ~JaxCustomMedium.from_eps_raw
      ~JaxCustomMedium.from_file
      ~JaxCustomMedium.from_hdf5
      ~JaxCustomMedium.from_hdf5_gz
      ~JaxCustomMedium.from_json
      ~JaxCustomMedium.from_nk
      ~JaxCustomMedium.from_orm
      ~JaxCustomMedium.from_tidy3d
      ~JaxCustomMedium.from_yaml
      ~JaxCustomMedium.generate_docstring
      ~JaxCustomMedium.get_jax_field_names
      ~JaxCustomMedium.get_sub_model
      ~JaxCustomMedium.get_submodels_by_hash
      ~JaxCustomMedium.get_tuple_group_name
      ~JaxCustomMedium.get_tuple_index
      ~JaxCustomMedium.grids
      ~JaxCustomMedium.help
      ~JaxCustomMedium.json
      ~JaxCustomMedium.make_inside_mask
      ~JaxCustomMedium.nk_model
      ~JaxCustomMedium.nk_to_eps_complex
      ~JaxCustomMedium.nk_to_eps_sigma
      ~JaxCustomMedium.parse_file
      ~JaxCustomMedium.parse_obj
      ~JaxCustomMedium.parse_raw
      ~JaxCustomMedium.plot
      ~JaxCustomMedium.schema
      ~JaxCustomMedium.schema_json
      ~JaxCustomMedium.sigma_model
      ~JaxCustomMedium.store_vjp
      ~JaxCustomMedium.to_file
      ~JaxCustomMedium.to_hdf5
      ~JaxCustomMedium.to_hdf5_gz
      ~JaxCustomMedium.to_json
      ~JaxCustomMedium.to_medium
      ~JaxCustomMedium.to_yaml
      ~JaxCustomMedium.tree_flatten
      ~JaxCustomMedium.tree_unflatten
      ~JaxCustomMedium.tuple_to_dict
      ~JaxCustomMedium.update_forward_refs
      ~JaxCustomMedium.updated_copy
      ~JaxCustomMedium.validate
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~JaxCustomMedium.freqs
      ~JaxCustomMedium.is_isotropic
      ~JaxCustomMedium.n_cfl
      ~JaxCustomMedium.permittivity
      ~JaxCustomMedium.conductivity
      ~JaxCustomMedium.eps_dataset
      ~JaxCustomMedium.interp_method
      ~JaxCustomMedium.subpixel
      ~JaxCustomMedium.name
      ~JaxCustomMedium.frequency_range
      ~JaxCustomMedium.allow_gain
      ~JaxCustomMedium.nonlinear_spec
   
   