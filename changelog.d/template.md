{% for section, _ in sections.items() %}
{% set section_name = definitions[section]["name"] if section else "" %}
{% if section_name %}
### {{ section_name }}

{% endif %}
{% for category, val in definitions.items() if category in sections[section] %}

### {{ definitions[category]["name"] }}

{% for text, values in sections[section][category].items() %}
- {{ text }}
{% endfor %}
{% endfor %}
{% endfor %}
{{ "\n" -}}
