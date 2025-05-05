import os
import numpy as np
WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
videos_path = f"{BASE_URL}\\videos"

loc3d = [2.589343,10.8,1.075566]
camVis = [ # xmin,xmax,ymin,ymax bbox seen in camera 2d image
        1910,
        463,
        1919,
        511
      ]

intrinsic = [
                [
                    1564.0372440427782,
                    0.0,
                    960.0
                ],
                [
                    0.0,
                    1564.0372440427782,
                    540.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]

extrinsic =  [
                [
                    0.7864581568462231,
                    0.6176435602595084,
                    -9.613776302819227e-09,
                    10.327416142239947
                ],
                [
                    0.10804354093484955,
                    -0.13757406212185255,
                    -0.9845811143291099,
                    0.9166555730907038
                ],
                [
                    -0.608120186141112,
                    0.774331847402166,
                    -0.17492864061224442,
                    22.39823681170598
                ]
            ]

camMat = [
        [
            28.85291554711405,
            76.31743930025183,
            -7.497532570772914,
            1681.1488638936025
        ],
        [
            -7.116666574115732,
            9.061795727051097,
            -72.9692684851526,
            604.0087623085254
        ],
        [
            -0.02715035970259303,
            0.03457110726667142,
            -0.007809929061953248,
            1.0
        ]]

homoMat = [
                [
                    28.85291554711405,
                    76.31743930025183,
                    1681.1488638936025
                ],
                [
                    -7.116666574115732,
                    9.061795727051097,
                    604.0087623085254
                ],
                [
                    -0.02715035970259303,
                    0.03457110726667142,
                    1.0
                ]
            ]

centerX = (camVis[0] + camVis[2])/2
centerY = (camVis[1] + camVis[3])/2





loc3d_m = np.array(loc3d + [1])

camMat = np.array(camMat)
inMat = np.array(intrinsic)
exMat = np.array(extrinsic)
homoMat = np.array(homoMat)


locMat = camMat @ loc3d_m
locMat /= locMat[2]

paraMat = inMat @ exMat @ loc3d_m
paraMat /= paraMat[2]




pixel_point = np.array([centerX, centerY, 1])


# Map image pixel -> map coordinates
mapped_point = homoMat @ pixel_point
mapped_point /= mapped_point[2]

# Optional: if you want to verify inverse works
reprojected_pixel = np.linalg.inv(homoMat) @ pixel_point
reprojected_pixel /= reprojected_pixel[2]

reprojected_pixel[0] = (reprojected_pixel[0] + 75)*14
reprojected_pixel[1] = (reprojected_pixel[1] + 41)*14


# -- Output --
print("3D loc: ", loc3d)
print("2D VIS (bbox center): ", (centerX, centerY))
print("2D MAP (via paraMat): ", (round(paraMat[0],1), round(paraMat[1],1)))
print("2D MAP (via camMat): ", (round(locMat[0],1), round(locMat[1],1)))
print("2D MAP (via homography): ", (round(mapped_point[0],1), round(mapped_point[1],1)))
print("Backprojected pixel (via inverse homography): ", (round(reprojected_pixel[0],1), round(reprojected_pixel[1],1)))

