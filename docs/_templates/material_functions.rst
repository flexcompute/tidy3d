Material library
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
