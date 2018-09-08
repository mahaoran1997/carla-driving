import time

import cv2
import scipy
import pygame
from extra import *

from drawing_tools import *

clock = pygame.time.Clock()


class ScreenManager(object):
    def __init__(self, load_steer=True):

        pygame.init()
        # Put some general parameterss
        self._render_iter = 0
        self._speed_limit = 50.0
        if load_steer:
            self._wheel = cv2.imread('./drive_interfaces/wheel.png')  # ,cv2.IMREAD_UNCHANGED)
            self._wheel = cv2.resize(self._wheel, (int(0.08 * self._wheel.shape[0]), int(0.08 * self._wheel.shape[1])))
        print(self._wheel.shape)

    # If we were to load the steering wheel load it


    # take into consideration the resolution when ploting
    # TODO: Resize properly to fit the screen ( MAYBE THIS COULD BE DONE DIRECTLY RESIZING screen and keeping SURFACES)

    def start_screen(self, resolution, aspect_ratio, scale=1):

        self._resolution = resolution

        self._aspect_ratio = aspect_ratio
        self._scale = scale

        size = (resolution[0] * aspect_ratio[0], resolution[1] * aspect_ratio[1])

        self._screen = pygame.display.set_mode((size[0] * scale, size[1] * scale), pygame.DOUBLEBUF)

        # self._screen.set_alpha(None)

        pygame.display.set_caption("Human/Machine - Driving Software")

        self._camera_surfaces = []

        for i in range(aspect_ratio[0] * aspect_ratio[1]):
            camera_surface = pygame.surface.Surface(resolution, 0, 24).convert()

            self._camera_surfaces.append(camera_surface)

    def paint_on_screen(self, size, content, color, position, screen_position):

        myfont = pygame.font.SysFont("monospace", size * self._scale, bold=True)

        position = (position[0] * self._scale, position[1] * self._scale)

        final_position = (position[0] + self._resolution[0] * (self._scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (self._scale * (screen_position[1]))))

        content_to_write = myfont.render(content, 1, color)

        self._screen.blit(content_to_write, final_position)

    def set_array(self, array, screen_position, position=(0, 0), scale=None):

        if scale == None:
            scale = self._scale

        if array.shape[0] != self._resolution[1] or array.shape[1] != self._resolution[0]:
            array = scipy.misc.imresize(array, [self._resolution[1], self._resolution[0]])

            # print array.shape, self._resolution

        final_position = (position[0] + self._resolution[0] * (scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (scale * (screen_position[1]))))

        # pygame.surfarray.array_colorkey(self._camera_surfaces[screen_number])
        self._camera_surfaces[screen_position[0] * screen_position[1]].set_colorkey((255, 0, 255))
        pygame.surfarray.blit_array(self._camera_surfaces[screen_position[0] * screen_position[1]],
                                    array.swapaxes(0, 1))

        camera_scale = pygame.transform.scale(self._camera_surfaces[screen_position[0] * screen_position[1]],
                                              (int(self._resolution[0] * scale), int(self._resolution[1] * scale)))

        self._screen.blit(camera_scale, final_position)

    def draw_wheel_on(self, steer, screen_position):

        cols, rows, c = self._wheel.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90 * steer, 1)
        rot_wheel = cv2.warpAffine(self._wheel, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # scale = 0.5
        position = (self._resolution[0] / 2 - cols / 2, int(self._resolution[1] / 1.5) - rows / 2)
        # print position


        wheel_surface = pygame.surface.Surface((rot_wheel.shape[1], rot_wheel.shape[0]), 0, 24).convert()
        # print array.shape, self._resolution

        # final_position = (position[0] + self._resolution[0]*(scale*(screen_number%3)),\
        #	position[1] + (self._resolution[1]*(scale*(screen_number/3))))


        # pygame.surfarray.array_colorkey(self._camera_surfaces[screen_number])
        wheel_surface.set_colorkey((0, 0, 0))
        pygame.surfarray.blit_array(wheel_surface, rot_wheel.swapaxes(0, 1))

        self._screen.blit(wheel_surface, position)

    # This one plot the nice wheel


    def plot_camera(self, sensor_data, screen_position=[0, 0]):

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)
            # print sensor_data.shape
        self.set_array(sensor_data, screen_position)

        pygame.display.flip()

    def plot_camera_steer(self, sensor_data, steer, screen_position=[0, 0]):

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)

        draw_path_on(sensor_data, 20, -steer * 10.0, (0, 255, 0))

        self.set_array(sensor_data, screen_position)

        pygame.display.flip()

    def plot_driving_interface(self, capture_time, sensor_data, \
                               action, direction, speed, screen_position=[0, 0], draw_wheel=False):

        start_to_print = time.time()
        steer = action.steer
        steer_noisy = action_noisy.steer
        acc = action.throttle
        brake = action.brake
        size_x, size_y, size_z = sensor_data.shape
        sensor_data = sensor_data[:, :, ::-1]
        # Define our fonts


        # draw_path_on(img, 10, -angle_steers*40.0)

        # draw_path_on(sensor_data, 20, -steer_noisy*20.0, (255, 0, 0))
        # draw_path_on(sensor_data, 20, -steer*20.0, (0, 255, 0))

        # draw_steer_on(sensor_data)

        draw_vbar_on(sensor_data, acc, int(1.5 * sensor_data.shape[0] / 8), (0, 255, 0))
        draw_vbar_on(sensor_data, brake, int(1.5 * sensor_data.shape[0] / 8) + 97, (255, 0, 0))
        initial_y_pos = size_x - size_x / 6 + 5
        self.set_array(sensor_data, screen_position)
        if draw_wheel:
            self.draw_wheel_on(steer, screen_position)

        self.paint_on_screen(size_x / 10, 'GAS', (0, 255, 0), (int(1.5 * sensor_data.shape[0] / 8) - 20, initial_y_pos),
                             screen_position)

        self.paint_on_screen(size_x / 10, 'BRAKE', (255, 0, 0),
                             (int(1.5 * sensor_data.shape[0] / 8) + 60, initial_y_pos), screen_position)
        # pygame.surfarray.blit_array(activation_surface, img_act)

        # pygame.display.flip()

        if direction == 4:
            text = "GO RIGHT"
            extraback = size_x / 7
        elif direction == 3:
            text = "GO LEFT"
            extraback = size_x / 13
        else:
            text = "GO STRAIGHT"
            extraback = int(size_x / 2.8)


            # direction_color = (255,0,0)

            # if ((self._render_iter)/10) % 2 != 0 and direction !=2.0:
            # direction_color = (0,255,0)
        if direction != 2:
            direction_pos = (size_y / 2 - size_x / 2 - extraback, size_x / 2 - size_x / 4)

            self.paint_on_screen(size_x / 6, text, (0, 255, 0), direction_pos, screen_position)

        self.paint_on_screen(size_x / 10, "Speed: %.2f" % speed, (0, 255, 0), (size_y / 2 - 55, 30), screen_position)

        pygame.display.flip()

        # pygame.image.save(self._screen, "capture2/img" + str(self._render_iter) +".png")

        self._render_iter += 1

    def plot3camrcnoise(self, sensor_data, \
                        steer, noise, difference, \
                        screen_number=0):

        # Define our fonts


        # draw_path_on(img, 10, -angle_steers*40.0)

        draw_path_on(sensor_data, 20, -steer * 20.0, (0, 255, 0))

        draw_path_on(sensor_data, 20, -noise * 20.0, (255, 0, 0))

        draw_path_on(sensor_data, 20, -difference * 20.0, (0, 0, 255))

        self.set_array(sensor_data, screen_number)

        pygame.display.flip()
