
class simulation_options(object):

    box_size = [4,4,4]
    size = 28.5
    corner_rad = 2.5
    target_dim = 90.0
    target_temp_1 = 1.75
    target_temp_2 = 1.50
    scale_factor = 5.0
    num_particles = 64

    def __init__(self, box_size = [4,4,4],size = 28.5, corner_rad = 2.5, target_dim = 90.0, target_temp_1 = 1.75, target_temp_2 = 1.5, scale_factor = 5.0, num_particles = 64):
        self.box_size = box_size
        self.size = size
        self.corner_rad = corner_rad
        self.target_dim = target_dim
        self.target_temp_1 = target_temp_1
        self.target_temp_2 = target_temp_2
        self.scale_factor = scale_factor
        self.num_particles = num_particles







