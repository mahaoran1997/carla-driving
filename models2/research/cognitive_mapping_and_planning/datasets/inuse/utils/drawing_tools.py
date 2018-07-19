from random import *
import colorsys
import pygame
import numpy as np
import matplotlib.pyplot as plt


from skimage import transform as trans


rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = trans.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def generate_ncolors(num_colors):

	color_pallet = []
	for i  in range(0,360, 360 / num_colors):

		hue = i
		saturation = 90 + float(randint(0,1000))/1000 * 10
		lightness = 50 + float(randint(0,1000))/1000 * 10

		color = colorsys.hsv_to_rgb(float(hue)/360.0,saturation/100,lightness/100) 


		color_pallet.append(color)

		#addColor(c);
	return color_pallet


def get_average_over_interval(vector,interval):

	avg_vector = []
	print 'interval',interval
	for i in range(0,len(vector),interval):

		initial_train =i
		final_train =i + interval
		

		avg_point = sum(vector[initial_train:final_train])/interval
		avg_vector.append(avg_point)

	return avg_vector


def  get_average_over_interval_stride(vector,interval,stride):

  avg_vector = []
  print 'interval',interval
  for i in range(0,len(vector)-interval,stride):

    initial_train =i
    final_train =i + interval
    

    avg_point = sum(vector[initial_train:final_train])/interval
    avg_vector.append(avg_point)

  return avg_vector







def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[int(row-sz):int(row+sz), int(col-sz-65):int(col+sz-65)] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function return teh lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)


# @param bar_intensity -> should go from -1 to 1

def draw_bar_on(img,bar_intensity,y_pos,color=(0,0,255)):


  bar_size = int(img.shape[1]/6 * bar_intensity)
  initial_x_pos = img.shape[1] - img.shape[1]/6
  #print bar_intensity

  for i in range(bar_size):
    if bar_intensity > 0.0:
      x = initial_x_pos - i
      img[y_pos , x] = color
      img[y_pos+1 , x] = color
      img[y_pos+2 , x] = color
    else: # negative bar  
      x = initial_x_pos + i
      img[y_pos, x ] = tuple([j/2 for j in color])
      img[y_pos+1, x ] = tuple([j/2 for j in color])
      img[y_pos+2, x ] = tuple([j/2 for j in color])
      print img[y_pos+2, x ]


def draw_vbar_on(img,bar_intensity,x_pos,color=(0,0,255)):


  bar_size = int(img.shape[1]/6 * bar_intensity)
  initial_y_pos = img.shape[0] - img.shape[0]/6
  #print bar_intensity

  for i in range(bar_size):
    if bar_intensity > 0.0:
      y = initial_y_pos - i
      for j in range(20):
        img[y , x_pos +j] = color

      #else: # negative bar  
      #   y = initial_y_pos + i
      #  img[y_pos, x ] = tuple([j/2 for j in color])
      #  img[y_pos+1, x ] = tuple([j/2 for j in color])
      #  img[y_pos+2, x ] = tuple([j/2 for j in color])
      #  print img[y_pos+2, x ]



def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

def grayscale_colormap(img,colormap):

  cmap = plt.get_cmap(colormap)
  #cmap = grayify_cmap(cmap)
  rgba_img = cmap(img)

  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img

