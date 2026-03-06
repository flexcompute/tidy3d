:html_theme.sidebar_secondary.remove:
{{ fullname | escape | underline}}

.. autoclass:: {{ fullname }}
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource
   :exclude-members: attrs, model_config, model_post_init, type

   {% block attributes %}
   {% if attributes %}
   {% set ns_attrs = namespace(items=[]) %}
   {% for item in attributes %}
   {% if item not in inherited_members
         and item != "attrs"
         and item not in ("model_config", "type")
         and not item.startswith("_") %}
   {% set ns_attrs.items = ns_attrs.items + [item] %}
   {% endif %}
   {%- endfor %}
   {% if ns_attrs.items %}
   .. rubric:: Attributes

   .. autosummary::
      {% for item in ns_attrs.items %}
        ~{{ fullname }}.{{ item }}
      {%- endfor %}
   {% endif %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   {% set ns_methods = namespace(items=[]) %}
   {% for item in methods %}
   {% if item not in inherited_members
         and item not in ("model_post_init",)
         and not item.startswith("_") %}
   {% set ns_methods.items = ns_methods.items + [item] %}
   {% endif %}
   {%- endfor %}
   {% if ns_methods.items %}
   .. rubric:: Methods

   .. autosummary::
       {% for item in ns_methods.items %}
            ~{{ fullname }}.{{ item }}
       {%- endfor %}
   {% endif %}
   {% endif %}
   {% endblock %}


   .. rubric:: Inherited Common Usage

   .. include:: ../_custom_autosummary/{{ fullname }}.rst
      :optional:
