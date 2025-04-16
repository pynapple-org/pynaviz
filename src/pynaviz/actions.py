import pygfx as gfx

def color_by(materials, **kwargs):
    """
    Color by a particular cmap

    materials:
        Dict of pygfx materials object
    array:
        some color value
    cmap:
        Default cmap (Jet)


    """
    for c in materials:
        materials[c].color = "red"

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