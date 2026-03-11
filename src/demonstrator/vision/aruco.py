# aruco_utils.py

import cv2
import numpy as np

def detect_aruco_markers_with_tracking(image, aruco_dict='DICT_5X5_50', marker_size_mm=50.0):
    """
    Angepasste Aruco-Marker Erkennung, die keine Texteinblendungen vornimmt.
    Gibt Tuple (pixel_mm_ratio, marker_corners) zurück. pixel_mm_ratio ist None, wenn kein Marker erkannt.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict_map = {
        'DICT_4X4_50':       cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100':      cv2.aruco.DICT_4X4_100,
        'DICT_5X5_50':       cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100':      cv2.aruco.DICT_5X5_100,
        'DICT_6X6_50':       cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100':      cv2.aruco.DICT_6X6_100,
        'DICT_7X7_50':       cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100':      cv2.aruco.DICT_7X7_100,
        'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    aruco_dict_type = aruco_dict_map.get(aruco_dict, cv2.aruco.DICT_5X5_50)
    aruco_dict_obj = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict_obj, parameters)

    # adaptive threshold + morphology to help detect faint markers
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    # detect in both binary and gray; pick whichever finds more
    corners_bin, ids_bin, _ = detector.detectMarkers(binary)
    corners_gray, ids_gray, _ = detector.detectMarkers(gray)
    if ids_bin is not None and len(ids_bin) > 0:
        marker_corners, marker_ids = corners_bin, ids_bin
    else:
        marker_corners, marker_ids = corners_gray, ids_gray

    pixel_mm_ratio = None
    if marker_ids is not None and len(marker_ids) > 0:
        # draw outlines on original image (no text)
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

        # compute ratio using first marker found
        # (we assume a square marker of known side length `marker_size_mm`)
        corners = marker_corners[0].reshape((4, 2))
        # Euclidean distance between two adjacent corners → pixel length
        side_px = np.linalg.norm(corners[0] - corners[1])
        pixel_mm_ratio = side_px / float(marker_size_mm)
    #print(f"###########################################################################")
    #rint(f"Marker gefunden! ID:{marker_ids}")
    return pixel_mm_ratio, marker_corners

def calculate_dimensions(box_xyxy, pixel_mm_ratio):
    """
    box_xyxy: [x1, y1, x2, y2]
    pixel_mm_ratio: pixels per millimeter (float)
    Rückgabe: (width_mm, height_mm) in Millimetern.
    Falls pixel_mm_ratio ist None, return (None, None).
    """
    if pixel_mm_ratio is None:
        return None, None
    x1, y1, x2, y2 = box_xyxy
    width_px  = x2 - x1
    height_px = y2 - y1
    return float(width_px) / pixel_mm_ratio, float(height_px) / pixel_mm_ratio
