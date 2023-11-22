tidy3d.plugins.adjoint.JaxMedium
================================

.. currentmodule:: tidy3d.plugins.adjoint

.. autoclass:: JaxMedium

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~JaxMedium.__init__
      ~JaxMedium.add_type_field
      ~JaxMedium.construct
      ~JaxMedium.copy
      ~JaxMedium.d_eps_map
      ~JaxMedium.dict
      ~JaxMedium.dict_from_file
      ~JaxMedium.dict_from_hdf5
      ~JaxMedium.dict_from_hdf5_gz
      ~JaxMedium.dict_from_json
      ~JaxMedium.dict_from_yaml
      ~JaxMedium.e_mult_volume
      ~JaxMedium.eps_comp
      ~JaxMedium.eps_complex_to_eps_sigma
      ~JaxMedium.eps_complex_to_nk
      ~JaxMedium.eps_diagonal
      ~JaxMedium.eps_model
      ~JaxMedium.eps_sigma_to_eps_complex
      ~JaxMedium.from_file
      ~JaxMedium.from_hdf5
      ~JaxMedium.from_hdf5_gz
      ~JaxMedium.from_json
      ~JaxMedium.from_nk
      ~JaxMedium.from_orm
      ~JaxMedium.from_tidy3d
      ~JaxMedium.from_yaml
      ~JaxMedium.generate_docstring
      ~JaxMedium.get_jax_field_names
      ~JaxMedium.get_sub_model
      ~JaxMedium.get_submodels_by_hash
      ~JaxMedium.get_tuple_group_name
      ~JaxMedium.get_tuple_index
      ~JaxMedium.help
      ~JaxMedium.json
      ~JaxMedium.make_inside_mask
      ~JaxMedium.nk_model
      ~JaxMedium.nk_to_eps_complex
      ~JaxMedium.nk_to_eps_sigma
      ~JaxMedium.parse_file
      ~JaxMedium.parse_obj
      ~JaxMedium.parse_raw
      ~JaxMedium.plot
      ~JaxMedium.schema
      ~JaxMedium.schema_json
      ~JaxMedium.sigma_model
      ~JaxMedium.store_vjp
      ~JaxMedium.to_file
      ~JaxMedium.to_hdf5
      ~JaxMedium.to_hdf5_gz
      ~JaxMedium.to_json
      ~JaxMedium.to_medium
      ~JaxMedium.to_yaml
      ~JaxMedium.tree_flatten
      ~JaxMedium.tree_unflatten
      ~JaxMedium.tuple_to_dict
      ~JaxMedium.update_forward_refs
      ~JaxMedium.updated_copy
      ~JaxMedium.validate
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~JaxMedium.n_cfl
      ~JaxMedium.permittivity
      ~JaxMedium.conductivity
      ~JaxMedium.name
      ~JaxMedium.frequency_range
      ~JaxMedium.allow_gain
      ~JaxMedium.nonlinear_spec
   
   