import numpy as np
import cv2
from utils import RSWrapper


if __name__ == '__main__':
    rs_wrapper = RSWrapper()
    for depth_image, color_image in rs_wrapper.iterate_over_frames():
        rainbow_colorized = cv2.cvtColor(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW), cv2.COLOR_BGR2RGB)
        cv2.imshow('Depth', rainbow_colorized)
        cv2.imshow('Color', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break