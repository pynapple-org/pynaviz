:html_theme.sidebar_secondary.remove:


Pynaviz : python neural analysis visualization
----------------------------------------------

|

.. grid:: 4
   :gutter: 2

   .. grid-item::
      .. image:: examples/example_dlc_pose_short.gif
         :width: 100%

   .. grid-item::
      .. image:: examples/example_head_direction_short.gif
         :width: 100%

   .. grid-item::
      .. image:: examples/example_lfp_short.gif
         :width: 100%

   .. grid-item::
      .. image:: examples/example_videos_short.gif
         :width: 100%


.. grid:: 1 1 2 2

   .. grid-item::

      .. grid:: auto

         .. button-ref:: installing
            :color: primary
            :shadow:

            Installing

         .. button-ref:: user_guide
            :color: primary
            :shadow:

            User guide

         .. button-ref:: examples
            :color: primary
            :shadow:

            Examples

         .. button-ref:: api
            :color: primary
            :shadow:

            API


Overview
--------

Pynaviz provides interactive, high-performance visualizations designed to work seamlessly
with Pynapple time series and video data. It allows synchronized exploration of neural signals
and behavioral recordings. It is built on top of `pygfx <https://pygfx.org/>`_, a modern GPU-based rendering engine.

There are two ways to use Pynaviz:

- `GUI <gui_reference.html>`_ — launch an interactive viewer from the command line (``pynaviz``) or from a Python
  script via :func:`scope`.  Drop in files, scrub through time, and arrange plots without
  writing any additional code.
- `Programmatic <user_guide.html>`_ — embed the individual plot widgets (``TsdWidget``, ``TsGroupWidget``, …)
  directly inside your own Qt application for tighter integration with custom pipelines.


Installation
------------

.. code-block:: bash

    $ pip install pynaviz[qt]

Please refer to the `Installation instructions <installing.html>`_ for more details.


Quick start
-----------

From the command line:

.. code-block:: bash

    $ pynaviz data.nwb recording.mp4 -l layout.json

From a Python script:

.. code-block:: python

    from pynaviz import scope
    scope({"lfp": tsdframe, "spikes": tsgroup})

See the `GUI reference <gui_reference.html>`_ for the full list of accepted file types,
``scope()`` input forms, and keyboard shortcuts.

.. toctree::
    :maxdepth: 1
    :hidden:

    Installing <installing>
    GUI reference <gui_reference>
    User guide <user_guide>
    Example gallery <examples>
    API <api>


Support
-------

This package is supported by the Center for Computational Neuroscience, in the Flatiron Institute of the Simons Foundation.

.. image:: _static/CCN-logo-wText.png
   :width: 200px
   :class: only-light
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/

.. image:: _static/logo_flatiron_white.svg
   :width: 200px
   :class: only-dark
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/