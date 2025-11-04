:html_theme.sidebar_secondary.remove:


pynaviz
=======

PYthon Neural Analysis VIZualization

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



Pynaviz provides interactive, high-performance visualizations designed to work seamlessly
with Pynapple time series and video data. It allows synchronized exploration of neural signals
and behavioral recordings. It is build on top of `pygfx <https://pygfx.org/>`_, a modern GPU-based rendering engine.

To install pynaviz, please refer to the `Installation instructions <installing.html>`_.

The simplest way to get started is to use the Qt-based graphical user interface (GUI) either from the command line :

.. code-block:: bash

    $ pip install pynaviz[qt]
    $ pynaviz


or from a Python script:

.. code-block:: python

    from pynaviz import scope
    scope("nwb_file.nwb") # replace with your NWB file path or a dictionnary of pynapple time series objects

.. toctree::
    :maxdepth: 1
    :hidden:

    Installing <installing>
    User guide <user_guide>
    Example gallery <examples>
    API <api>


|

See the `User Guide <user_guide.html>`_ for more details on how to use pynaviz.




Support
~~~~~~~

This package is supported by the Center for Computational Neuroscience, in the Flatiron Institute of the Simons Foundation

.. image:: _static/CCN-logo-wText.png
   :width: 200px
   :class: only-light
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/

.. image:: _static/logo_flatiron_white.svg
   :width: 200px
   :class: only-dark
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/

