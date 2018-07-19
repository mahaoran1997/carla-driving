from planner import Waypointer

import os

city_name ='carla_0'
dir_path = os.path.dirname(__file__)
waypointer = Waypointer(dir_path+'planner/' + city_name + '.txt',\
	dir_path+'planner/' + city_name + '.png')


#planner.get_next_waypoints(planner.make_world_map([1197,115]),(1,0,0),planner.make_world_map([121,227]),(1,0,0))

#planner.get_next_waypoints(planner.make_world_map([111,949]),(0,-1,0),planner.make_world_map([110,160]),(1,0,0))

waypointer.get_next_waypoints(waypointer.make_world_map([77,200]),(0,-1,0),waypointer.make_world_map([399,72]),(1,0,0))


