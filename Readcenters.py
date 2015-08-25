#Read in f_center. Original by Ting, heavy modifications by M.G.
__Author = 'M.G.'

def read_centers(f_center):
    '''simple file open + split & strip'''
    fc = open(f_center, 'r')
    lines = fc.readlines()[2:]

    positions = []
    type = []
    for line in lines:
        l = line.strip().split()
        pos = [float(l[1])*1.0, float(l[2])*1.0, float(l[3])*1.0]
        positions.append(pos)
        type.append(l[0])

    return type, positions