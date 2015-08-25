class bead():
  '''This is a coarse grained bead'''
  '''Attributes'''
  def __init__(self, position, beadtype='notype', mass=1.0, body=-1, image=[0, 0, 0]):
    self.beadtype = beadtype
    self.mass = mass
    self.position = position
    self.body = body
    self.image = image
  
