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


Installation
------------

.. code-block:: bash

    $ pip install pynaviz[qt]

Please refer to the `Installation instructions <installing.html>`_ for more details.


Quick start
-----------

From the command line
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ pynaviz [files ...] [-l layout.json] [-f FORMAT]

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Example
     - Argument
     - Notes
   * - ``$ pynaviz``
     - *(no arguments)*
     - Opens an empty viewer
   * - ``$ pynaviz data.nwb``
     - ``files``
     - One or more ``.nwb`` files; objects unpacked individually
   * - ``$ pynaviz data.npz``
     - ``files``
     - One or more ``.npz`` files; must contain a single pynapple object each
   * - ``$ pynaviz recording.mp4``
     - ``files``
     - One or more video files (``.mp4``, ``.avi``, ``.mov``, ``.mkv``)
   * - ``$ pynaviz data.nwb recording.mp4``
     - ``files``
     - Multiple files of different types can be mixed
   * - ``$ pynaviz recording.plx``
     - ``files``
     - Ephys file; format auto-detected via ``nap.EphysReader``
   * - ``$ pynaviz rec/``
     - ``files``
     - Directory; auto-detected as NeuroScopeIO if ``.dat`` + ``.xml`` are present
   * - ``$ pynaviz rec/ -f NeuroScopeIO``
     - ``files`` + ``-f``
     - Directory with explicit Neo IO format
   * - ``$ pynaviz recording.plx -f PlexonIO``
     - ``files`` + ``-f``
     - Ephys file with explicit format
   * - ``$ pynaviz -l layout.json``
     - ``-l`` / ``--layout``
     - Restore a previously saved layout (``.json``)
   * - ``$ pynaviz data.nwb -l layout.json``
     - ``files`` + ``-l``
     - Load files and restore layout simultaneously

From a Python script
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pynaviz import scope

The :func:`scope` function accepts many input types:

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Example
     - Input type
     - Notes
   * - ``scope({"lfp": tsdframe, "spikes": tsgroup})``
     - ``dict``
     - Keys become display names in the variable panel
   * - ``scope([tsdframe, tsgroup, interval_set])``
     - ``list`` / ``tuple``
     - Names inferred from class (``TsdFrame``, ``TsGroup``, …)
   * - ``scope(tsgroup)``
     - ``nap.TsGroup``
     - Collection of spike trains. Same for all pynapple objects (``Tsd``, ``TsdFrame``, …)
   * - ``scope(nap.load_file("data.nwb"))``
     - ``nap.NWBFile``
     - All contained objects unpacked individually
   * - ``scope(nap.EphysReader("rec/", format="NeuroScopeIO"))``
     - ``nap.EphysReader``
     - All contained objects unpacked individually
   * - ``scope("data.nwb")``
     - ``str`` / ``pathlib.Path`` — ``.nwb``
     - Loaded via pynapple, objects unpacked
   * - ``scope("data.npz")``
     - ``str`` / ``pathlib.Path`` — ``.npz``
     - Must contain a single pynapple object
   * - ``scope("recording.mp4")``
     - ``str`` / ``pathlib.Path`` — video
     - ``.mp4``, ``.avi``, ``.mov``, ``.mkv`` supported
   * - ``scope("recording.plx")``
     - ``str`` / ``pathlib.Path`` — ephys file
     - Loaded via ``nap.EphysReader``; format auto-detected
   * - ``scope("rec/")``
     - ``str`` / ``pathlib.Path`` — directory
     - Directory passed to ``nap.EphysReader``; auto-detects NeuroScopeIO

See the `User Guide <user_guide.html>`_ for more details.


Keyboard shortcuts
------------------

Global
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Shortcut
     - Action
   * - :kbd:`Space`
     - Play / pause
   * - :kbd:`Ctrl+S`
     - Save layout
   * - :kbd:`Ctrl+O`
     - Load layout

Per-dock (active when the mouse is over the canvas)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Shortcut
     - Action
   * - :kbd:`r`
     - Reset view
   * - :kbd:`←` / `→`
     - Pan left / right by one page
   * - :kbd:`y`
     - Lock / unlock y-axis
   * - :kbd:`x`
     - Lock / unlock x-axis
   * - :kbd:`Ctrl+← / Ctrl+→`
     - Jump to previous / next superposed epoch (requires an ``IntervalSet`` overlay)
   * - :kbd:`i` / :kbd:`d`
     - Increase / decrease contrast (TsdFrame) or marker size (TsGroup)
   * - :kbd:`n` / :kbd:`p`
     - Jump to next / previous interval or timestamp (IntervalSet & Ts)

.. toctree::
    :maxdepth: 1
    :hidden:

    Installing <installing>
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