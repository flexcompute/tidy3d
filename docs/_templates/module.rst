:html_theme.sidebar_secondary.remove:
{{ fullname | escape | underline}}

.. autoclass:: {{ fullname }}
   :members:
   :inherited-members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
      {% for item in attributes %}
        {{ item }}
      {%- endfor %}
      {% endif %}
      {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
       :toctree:
       {% for item in methods %}
            {{ item }}
       {%- endfor %}
       {% endif %}
       {% endblock %}

