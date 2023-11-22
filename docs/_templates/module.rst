{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :member-order: bysource

   { {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
      :toctree:
       {% for item in attributes %}
          {{ item }}
       {%- endfor %}
       {% endif %}
       {% endblock %}