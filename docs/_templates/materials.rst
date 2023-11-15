Material library
{{ underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}

   .. autosummary::
      :toctree: {{ objname }}
      :template: material_class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}