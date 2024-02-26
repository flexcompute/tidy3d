:html_theme.sidebar_secondary.remove:
{{ fullname | escape | underline}}

{{ fullname }}
{{ attributes }}
{{ methods }}
{{ members }}
{{ inherited_members }}
hey
{{ custom_documentation_list }}

.. autoclass:: {{ fullname }}
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
      {% for item in attributes %}
          {% if item not in inherited_members %}
            {{ item }}
          {% endif %}
          {% if item in attributes.custom_documentation_list %}
            {{ item }}
          {% endif %}
      {%- endfor %}
      {% endif %}
      {% endblock %}


   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
       :toctree:
       {% for item in methods %}
          {% if item not in inherited_members %}
            {{ item }}
          {% endif %}
       {%- endfor %}
       {% endif %}
       {% endblock %}

   {% block custom_documentation %}
   {% if "custom_documentation_list" in members %}
   {{hey}}
   .. rubric:: Common Usage

   .. autosummary::
       :toctree:
       {% for item in members %}
        {{ item }}
       {%- endfor %}
       {% endif %}
       {% endblock %}
