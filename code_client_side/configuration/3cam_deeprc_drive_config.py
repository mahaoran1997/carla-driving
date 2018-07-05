

class configDrive:

  def __init__(self):

    #self.experiment_name =''

    self.path = "../Desktop/" # If path is set go for it , if not expect a name set
    self.resolution = [320,240]
    self.noise = "None"
    self.type_of_driver = "Human"
    self.interface = "DeepRC"
    self.show_screen = False
    self.number_screens = 2
    self.cameras_to_plot = {0:0}
    #self.cameras_to_plot = {0:0,1:1,2:2}
    self.middle_camera = 0
    self.scale_factor = 2
    self.image_cut =[80,220] # This is made from top to botton
    self.augment_left_right = True
    self.camera_angle = 30.0
    self.plot_vbp = True
