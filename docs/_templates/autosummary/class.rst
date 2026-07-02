{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% set public_methods = methods | reject("equalto", "__init__") | list %}

{% block attributes %}
{% if attributes %}
Attributes table
~~~~~~~~~~~~~~~~

.. autosummary::
{% for item in attributes %}
    ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if public_methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
{% for item in public_methods %}
    ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
Attributes
~~~~~~~~~~

{% for item in attributes %}

.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if public_methods %}
Methods
~~~~~~~

{% for item in public_methods %}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endfor %}

{% endif %}
{% endblock %}
