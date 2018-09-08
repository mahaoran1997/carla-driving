import sys

sys.path.append('../utils')
from drawing_tools import *

import cv2 as cv

import os


def write_image(img, position, speed_ms, angle_steers, acc, config, number_of_screens, text, index, color=(0, 0, 255)):
    # img = img.transpose((1,2,0))
    write_images_to = 'output_images'

    # print img
    # print speed_ms
    # print angle_steers*30
    draw_path_on(img, speed_ms, angle_steers, color)
    draw_bar_on(img, acc, img.shape[0] / 8, color)

    # global canvas_size
    canvas_size = (config.image_size[1] * number_of_screens, config.image_size[0])
    # imgW = img.width
    # imgH = img.height

    canvas_image_size = (w, h, channels) = (config.image_size[0], config.image_size[1] * number_of_screens, 3)

    img_canvas = np.zeros(canvas_image_size, np.uint8)
    # img_canvas = img_canvas.transpose(1,0,2)
    # cv2.imshow('result', img), cv2.waitKey(0)

    # vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
    # print img_canvas.shape
    font = cv.FONT_HERSHEY_PLAIN
    # print text

    str_text = ','.join(str(int(e)) for e in text)

    cv.putText(img, str_text, (40, 40), font, 3, (255, 0, 0), 2)

    img_canvas[:, (position * config.image_size[1]):((position + 1) * config.image_size[1]), :] = img

    # img_canvas = img_canvas.transpose(1,0,2)


    if not os.path.exists(write_images_to):
        os.makedirs(write_images_to)
    cv.imwrite(write_images_to + '/' + str(index) + '.png', img_canvas)

# cv.NamedWindow("Adas Window", cv.CV_WINDOW_AUTOSIZE)
# cv.ResizeWindow("Adas Window", canvas_size[0] , canvas_size[1])
# cv.ShowImage("Adas Window", vis0 )
# cv.WaitKey(1)



# pygame.surfarray.blit_array(camera_surface_vec[position], img.swapaxes(0,1))
# camera_surface_2x = pygame.transform.scale2x(camera_surface_vec[position])
# screen.blit(camera_surface_2x, (position*2*256,0))
# pygame.display.flip()
