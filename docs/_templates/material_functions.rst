Material library
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   {{ objname }}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
