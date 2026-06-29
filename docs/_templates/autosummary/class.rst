{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
Methods
~~~~~~~

{% for item in methods %}
{%- if item != '__init__' %}
.. automethod:: {{ [objname, item] | join(".") }}
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
