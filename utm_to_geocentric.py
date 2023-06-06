import glob
import sat_utils
import numpy as np
import argparse
from bundle_adjust import geo_utils

def utm_to_geocentric(inFile, outFile, zonestring):
    print('-------------------------')
    print('inFile: ', inFile)
    print('outFile: ', outFile)
    pts3d = np.loadtxt(inFile, dtype='float')
    
    easts = pts3d[:,0]
    norths = pts3d[:,1]
    alts = pts3d[:,2]

    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)

    x, y, z = sat_utils.latlon_to_ecef_custom(lats, lons, alts)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    z = z[:, np.newaxis]
    pts3d = np.hstack((x, y, z))

    np.savetxt(outFile, pts3d, fmt="%lf",delimiter=' ') #,header ='first,second,trid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, required=True,
                        help='file_dir')
    parser.add_argument('--zonestring', type=str, default='17',
                        help='zonestring')
    args = parser.parse_args()

    densedepth_files = sorted(glob.glob(args.file_dir + "/*_3DPts.txt"))

    for t, densedepth_file in enumerate(densedepth_files):
        print('--------------', t, densedepth_file, '--------------')
        inFile = densedepth_file
        outFile = densedepth_file[:-4] + "_ecef.txt"
        utm_to_geocentric(inFile, outFile, args.zonestring)

