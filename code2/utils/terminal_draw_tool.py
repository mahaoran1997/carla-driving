
import curses
import time
import os

curses_started = False

def start_curses():
	global stdscr
	stdscr = curses.initscr()
	stdscr.resize(200,200)
	curses.start_color()
	curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
	curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
	curses.use_default_colors()

def clear():

	stdscr.clear()


def print_to_position(x,y,string,c_number=1):
	stdscr.addstr(x, y,string,curses.color_pair(c_number))
	stdscr.refresh()
    
def draw_experiment_name():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	print_to_position(0,0,dir_path)
    

def draw_training_percentage(percentage,elapse_time=0.0):

	initial_string = 'Training Execution : '
	print_to_position(0,0,initial_string)
	print_to_position(0,len(initial_string),'['+'-'*100+']')
	print_to_position(0,len(initial_string)+1,'='*percentage)

	time_str = ' Time: '
	print_to_position(0,len(initial_string)+len('['+'-'*100+']')+1,time_str)

	print_to_position(0,len(initial_string)+len('['+'-'*100+']')+len(time_str) +1,str(elapse_time))

	if percentage<100:
		print_to_position(0,len(initial_string)+ percentage,'>')


def draw_line(line_number,var_names,var_values,color_vec):

	xposition = -1  # Just to start at begining
	for i in range(len(var_names)):
		name = var_names[i]		
		print_to_position(line_number,xposition+1,name)
		xposition += len(name)
		value = var_values[i]
		print_to_position(line_number,xposition+1,value,color_vec[i])
		xposition += len(value)


def draw_line_val(line_number,var_values,color_vec,spacing=1):

	xposition = -1  # Just to start at begining
	for i in range(len(var_values)):

		value = var_values[i]
		print_to_position(line_number,xposition+1,value,color_vec[i])
		xposition += len(value) +spacing


def draw_vector_four_col(initial_line,var_values1,var_values2,var_values3,var_values4):
	ScreenH, ScreenW = stdscr.getmaxyx()
	print ScreenH
	for i in range(0,min(var_values1.shape[0],ScreenH-4)):

		draw_line_val(initial_line + i,[str(var_values1[i]),str(var_values2[i]),str(var_values3[i]),str(var_values4[i])],[1,2,3,1])


def draw_vector_five_col(initial_line,var_values1,var_values2,var_values3,var_values4,var_values5):
	ScreenH, ScreenW = stdscr.getmaxyx()
	print ScreenH
	for i in range(0,min(var_values1.shape[0],ScreenH-4)):

		draw_line_val(initial_line + i,[str(var_values1[i]),str(var_values2[i]),str(var_values3[i]),str(var_values4[i]),str(var_values5[i])],[1,2,3,1,1])



def draw_vector_six_col(initial_line,var_values1,var_values2,var_values3,var_values4,var_values5,var_values6):
	ScreenH, ScreenW = stdscr.getmaxyx()
	print ScreenH
	for i in range(0,min(var_values1.shape[0],ScreenH-4)):

		draw_line_val(initial_line + i,[str(var_values1[i]),str(var_values2[i]),str(var_values3[i]),str(var_values4[i]),str(var_values5[i]),str(var_values6[i])],[1,2,3,1,1,1])

def draw_vector_n_col(initial_line,var_values):
	ScreenH, ScreenW = stdscr.getmaxyx()
	#print ScreenH
	
	for i in range(0,min(var_values[0].shape[0],ScreenH-4)):

		vector_to_draw = []
		color_vec = []
		#print 'len ',len(var_values)
		#print 'len2 ',len(var_values[0])
		for j in range(len(var_values)):
			#print 'i,j ',i,j
			#print 'shape :',var_values[j].shape
			vector_to_draw.append("%.4f"%(var_values[j][i]))
			color_vec.append(1)

		color_vec[1] = 2
		color_vec[2] = 3


		draw_line_val(initial_line + i,vector_to_draw,color_vec)


#draw_training_percentage(10)

#draw_line(1,['Epoch: ',' Step: ',' Images P/Sec: '],[50.0123,1231323,324.02],[1,1,1])
#draw_line(2,['Best Train: ',' Best Validation: ',' Current Train: '],[0.0123,0.0323,0.1],[2,2,1])





#time.sleep(5)

