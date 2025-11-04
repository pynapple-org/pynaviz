.. _api_ref:

API reference
=============

.. rubric:: Base plot

.. currentmodule:: pynaviz.base_plot

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    PlotTsd
    PlotTsdFrame
    PlotTs
    PlotTsGroup

.. currentmodule:: pynaviz.audiovideo.video_plot

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    PlotTsdTensor
    PlotVideo

.. rubric:: Custom controllers

.. currentmodule:: pynaviz.controller

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    SpanController
    SpanYLockController
    GetController

.. rubric:: Audio and Video Manipulation

Classes allowing random access to video and audio frames, with ``pynapple``-like ``get`` syntax or by indexing.

.. currentmodule:: pynaviz.audiovideo

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    AudioHandler
    VideoHandler


.. rubric:: Pynaviz

.. currentmodule:: pynaviz

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    base_plot
    audiovideo
    controller
    events
    interval_set
    plot_manager
    utils
    synchronization_rules

.. rubric:: Qt

.. currentmodule:: pynaviz.qt

Module including Qt widgets with selection, sorting and coloring by metadata functionalities.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    mainwindow
    widget_plot
    widget_menu
    widget_list_selection
    tsdframe_selection
    interval_sets_selection