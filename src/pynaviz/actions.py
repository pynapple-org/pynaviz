import pygfx as gfx
import matplotlib.cm as cm



def color_by(materials, values, cmap_name='jet'):
    """
    Color by a particular cmap

    materials:
        Dict of pygfx materials object
    values:
        pandas.Series
    cmap:
        The color map. Default is "jet".
        See https://matplotlib.org/stable/gallery/color/colormap_reference.html


    """
    cmap = cm.get_cmap(cmap_name)
    values = values - values.min()
    values = values / values.max()
    for c in materials:
        materials[c].color = cmap(values[c])

def sort_by(geometries, values):
    """
    Sort by values

    geometries:
        Dict of pygfx geometries object
    values:
        Dict of index -> values to sort

    """
    # This should do the sorting
    print(values)

def group_by(geometries, values):
    """
    Group by values
    """
    pass