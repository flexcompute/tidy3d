:html_theme.sidebar_secondary.remove:
{{ fullname | escape | underline}}

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