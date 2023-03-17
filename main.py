# Author: Leonardo Rossi Leão
# E-mail: leonardo.leao@cnpem.br

from datetime import datetime
from actions.archiver import Archiver
from actions.plot import Plot3D, Plot2D

# Request variables from archiver
# pvs_hls_level = [
#     "TU-01C:SS-HLS-Ax16SE4:Level-Mon",
#     "TU-01C:SS-HLS-Ax18SE5:Level-Mon",
#     "TU-03C:SS-HLS-Ax24SW1:Level-Mon",
#     "TU-04C:SS-HLS-Ax27SW2:Level-Mon",
#     "TU-05C:SS-HLS-Ax29SW3:Level-Mon",
#     "TU-06C:SS-HLS-Ax31SW4:Level-Mon",
#     "TU-06C:SS-HLS-Ax33SW5:Level-Mon",
#     "TU-08C:SS-HLS-Ax39NW1:Level-Mon",
#     "TU-09C:SS-HLS-Ax42NW2:Level-Mon",
#     "TU-10C:SS-HLS-Ax44NW3:Level-Mon",
#     "TU-11C:SS-HLS-Ax46NW4:Level-Mon",
#     "TU-11C:SS-HLS-Ax48NW5:Level-Mon",
#     "TU-13C:SS-HLS-Ax54NE1:Level-Mon",
#     "TU-14C:SS-HLS-Ax57NE2:Level-Mon",
#     "TU-15C:SS-HLS-Ax59NE3:Level-Mon",
#     "TU-16C:SS-HLS-Ax01NE4:Level-Mon",
#     "TU-17C:SS-HLS-Ax04NE5:Level-Mon",
#     "TU-18C:SS-HLS-Ax09SE1:Level-Mon",
#     "TU-19C:SS-HLS-Ax12SE2:Level-Mon",
#     "TU-20C:SS-HLS-Ax14SE3:Level-Mon",
# ]

pvs_hls_level = Archiver.getPVs(["*hls*setup*level"])

# Request data from HLS sensors
ini = datetime(2023, 2, 28, 0, 0)
end = datetime(2023, 3, 3, 0, 0)
hls_level = Archiver.request(pvs_hls_level, ini, end, 200)

#Plot3D.dynamic_plot(hls_level, save_all_figures=False)
Plot2D.static_plot(hls_level, diff=True)