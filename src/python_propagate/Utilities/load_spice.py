import spiceypy as spice

#TODO Have the user indicate which spice files to download

def load_spice():
    spice.furnsh('data/naif0011.tls')
    spice.furnsh('data/earth_000101_250421_250124.bpc')
    # spice.furnsh('data/pck00010.tpc')
    # spice.furnsh('data/de430.bsp')

def unload_spice():
    spice.unload('data/naif0011.tls')
    spice.unload('data/earth_000101_250421_250124.bpc')
    # spice.unload('data/pck00010.tpc')
    # spice.unload('data/de430.bsp')