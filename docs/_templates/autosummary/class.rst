{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{# Attribute descriptions come from the class docstring's ``Attributes`` section,
   rendered by napoleon inside ``.. autoclass::`` above. Summary tables and a second
   ``autoattribute`` loop would register attributes twice (or dangle on undocumented
   ones) and fail the strict (-W) build, so only inline Methods are listed below. #}

{% set public_methods = methods | reject("equalto", "__init__") | list %}

{% block methods_documentation %}
{% if public_methods %}
Methods
~~~~~~~

{% for item in public_methods %}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endfor %}

{% endif %}
{% endblock %}
