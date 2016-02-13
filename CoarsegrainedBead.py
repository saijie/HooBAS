class bead(object):
  """
  This is a coarse grained bead
  """
  def __init__(self, position, beadtype='notype', mass=1.0, body=-1, image=None, charge = 0.0, diameter = 0.0):
    self.beadtype = beadtype
    self.mass = mass
    self.position = position
    self.body = body
    if image is None :
        self.image = [0, 0, 0]
    else:
        self.image = image
    self.charge = charge
    self.diameter = diameter
  
