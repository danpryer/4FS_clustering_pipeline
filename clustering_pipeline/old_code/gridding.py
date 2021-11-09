# A class to define the "template" grid used in a power spectrum calculation.
# The object is basically a gridded box and has properties like volume, number of cells,
# array of grid point positions in each direction, nyquist frequency etc.
class set_grid:
    def __init__(self, box_points, x_data, y_data, z_data, x0, y0, z0):
        if float(np.log2(box_points)).is_integer() == False:
            raise ValueError('Number of box points must be an integer power of 2.')
        else:
            self.box_points = box_points
            self.nx = box_points
            self.ny = box_points
            self.nz = box_points
            self.Nc = self.nx*self.ny*self.nz  # number of cells

            self.x_min = np.floor(min(x_data))
            self.x_max = np.ceil(max(x_data))
            self.Lx = self.x_max - self.x_min  # total side length
            self.lx = self.Lx / box_points     # cell side length
            self.x_points = np.linspace(self.x_min, self.x_max, box_points)

            self.y_min = np.floor(min(y_data))
            self.y_max = np.ceil(max(y_data))
            self.Ly = self.y_max - self.y_min
            self.ly = self.Ly / box_points
            self.y_points = np.linspace(self.y_min, self.y_max, box_points)

            self.z_min = np.floor(min(z_data))
            self.z_max = np.ceil(max(z_data))
            self.Lz = self.z_max - self.z_min
            self.lz = self.Lz / box_points
            self.z_points = np.linspace(self.z_min, self.z_max, box_points)

            self.Vol = self.Lx * self.Ly * self.Lz    # total box volume
            self.Vc = self.Vol / self.Nc              # cell volume

            # origin coords
            self.x0 = x0
            self.y0 = y0
            self.z0 = z0

            # nyquist frequencies
            self.knx = np.pi / self.lx
            self.kny = np.pi / self.ly
            self.knz = np.pi / self.lz
