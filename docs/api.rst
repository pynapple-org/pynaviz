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
    PlotTsdTensor
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
    video
    controller
    events
    interval_set
    plot_manager
    qt_item_models
    widget_plot
    widget_menu
    utils
    synchronization_rules