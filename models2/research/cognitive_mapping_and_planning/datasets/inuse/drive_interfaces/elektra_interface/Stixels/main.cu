
//Including the ZED Camera
#include <sl/Camera.hpp>

#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>

#include <iostream>
#include <numeric>
#include <ctime>
#include <sys/types.h>
#include <sys/time.h>
#include <stdint.h>
#include <linux/limits.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include "Stixels.hpp"
#include "RoadEstimation.h"
#include "configuration.h"
#include "disparity_method.h"
#include <unistd.h>
//Including OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define OVERWRITE	true

cv::Mat slMat2cvMat(sl::Mat& input);

void startZED();

int clientSocket, portNum, nBytes, old_car_speed=0, new_car_speed = 0;
char buffer[]="H";
struct sockaddr_in serverAddr;
socklen_t addr_size;

  									/*Create UDP socket*/
void SaveStixels(std::vector<Section> *stixels, const int real_cols, const char *fname) {
	std::ofstream fp;
	fp.open (fname, std::ofstream::out | std::ofstream::trunc);
	//fp << "Writing this to a file.\n";
	if(fp.is_open()) {
		for(size_t i = 0; i < real_cols; i++) {
			std::vector<Section> sections_vec = stixels[i];
			for(size_t j = 0; j < sections_vec.size(); j++) {
				Section section = sections_vec.at(j);
				fp << section.type << "," << section.vB << "," << section.vT << "," << section.disparity << ";";
			}
			// Column finished
			fp << std::endl;
		}
		fp.close();
	} else {
		std::cerr << "Counldn't write file: " << fname << std::endl;
	}
}

void SaveStixels(Section *stixels, const int real_cols, const int max_segments, const char *fname) {
	std::ofstream fp;
	fp.open (fname, std::ofstream::out | std::ofstream::trunc);
	//fp << "Writing this to a file.\n";
	
	if(fp.is_open()) {
		for(size_t i = 0; i < real_cols; i++) {
			for(size_t j = 0; j < max_segments; j++) {
				Section section = stixels[i*max_segments+j];
				if(section.type == -1) {
					break;
				}
								
				// If disparity is 0 it is sky
				if(section.type == OBJECT && section.disparity < 1.0f) {
					section.type = SKY;
				}
				fp << section.type << "," << section.vB << "," << section.vT << "," << section.disparity << ";";
			}
			// Column finished
			fp << std::endl;
		}
		fp.close();
	} else {
		std::cerr << "Counldn't write file: " << fname << std::endl;
	}

}

int MoveCar(Section *stixels, int* toproad_row, const int real_cols, const int max_segments, char* buffer, int old_car_speed) {
	
	
	int level_height = (*toproad_row)/3;	
	int ground_area=0, ground_area_level2=0, ground_area_level3=0, section_area = 5*50*level_height, section_area_level2 = 5*26*level_height, section_area_level3 = 5*14*level_height;
	//std::cout<<"Real cols" <<real_cols << ";" <<  std::endl;
	//std::cout<<"level height"<<level_height<<std::endl;
	for(size_t i =(real_cols/2 - 25) ; i < (real_cols/2 + 25); i+=1) {

		for(size_t j=0;j<max_segments;j++) {
		
			Section section = stixels[i*max_segments+j];
			//fp<<section.type<<","<<section.vB<<","<<section.vT<<","<<section.disparity<<";"<<std::endl;
			if(section.vT<= level_height && section.type == GROUND)
				ground_area+=5*(section.vT-section.vB);
				else if (section.vB < level_height && section.vT >=level_height && section.type == GROUND)
					ground_area+=5*(level_height-section.vB);		
		}			
	}
	//fp.close();
	std::cout<<"Ratio at level 1"<<(float)ground_area/(float)section_area<<std::endl;
	if((float)ground_area/(float)section_area > 0.95){
		new_car_speed=10;
		for(size_t i =(real_cols/2 - 13) ; i < (real_cols/2 + 13); i+=1) {

					
			for(size_t j=0;j<max_segments;j++) {
		
				Section section2 = stixels[i*max_segments+j];
				if(section2.vB>=level_height && section2.vT<=2*level_height && section2.type == GROUND)
					ground_area_level2+=5*(section2.vT-section2.vB);
					else if(section2.vB<=level_height && section2.vT<=2*level_height && section2.vT>=level_height && section2.type == GROUND)
						ground_area_level2+=5*(section2.vT-level_height);
						else if(section2.vB>=level_height && section2.vT>=2*level_height && section2.type == GROUND)
							ground_area_level2+=5*(2*level_height-section2.vB);
							else if(section2.vB<=level_height && section2.vT>=2*level_height && section2.type == GROUND)
								ground_area_level2+=5*(level_height);		
			//std::cout<<ground_area_level2<<"is value of ground_area_level2"<<std::endl;
			}			
		}
		std::cout<<"Ratio at level 2"<<(float)ground_area_level2/(float)section_area_level2<<std::endl;
		if((float)ground_area_level2/(float)section_area_level2 > 0.95){
			new_car_speed=20;
			for(size_t i =(real_cols/2 - 7) ; i < (real_cols/2 + 7); i+=1) {

				Section section3;		
				for(size_t j=0;j<max_segments;j++) {
		
					section3 = stixels[i*max_segments+j];
					if(section3.vB>=2*level_height && section3.vT<=3*level_height && section3.type == GROUND)
						ground_area_level3+=5*(section3.vT-section3.vB);
					else if(section3.vB<=2*level_height && section3.vT<=3*level_height && section3.vT>=2*level_height && section3.type == GROUND)
						ground_area_level3+=5*(section3.vT-2*level_height);
						else if(section3.vB>=2*level_height && section3.vT>=3*level_height && section3.type == GROUND)
							ground_area_level3+=5*(3*level_height-section3.vB);
							else if(section3.vB<=2*level_height && section3.vT>=3*level_height && section3.type == GROUND)
								ground_area_level3+=5*(level_height);				
				}			
			}
			std::cout<<"Ratio at level3"<<(float)ground_area_level3/(float)section_area_level3<<std::endl;
			if((float)ground_area_level3/(float)section_area_level3 > 0.85)
				new_car_speed=30;
		}	
	}
	else new_car_speed=0;

	int section1_count=0, section2_count=0, section4_count=0, section5_count=0, new_rotation=0,section2_area= 5*(((real_cols/2)-25)/2)*level_height;

	for(size_t i=(real_cols/2 - 25)/2 ; i < (real_cols/2 - 25); i+=1 ){
		for(size_t j=0;j<max_segments;j++) {
		
			Section section = stixels[i*max_segments+j];
			//fp<<section.type<<","<<section.vB<<","<<section.vT<<","<<section.disparity<<";"<<std::endl;
			if(section.vT<= level_height && section.type == OBJECT)
				section2_count+=5*(section.vT-section.vB);
				else if (section.vB < level_height && section.vT >=level_height && section.type == OBJECT)
					section2_count+=5*(level_height-section.vB);		
		}	
	}
	std::cout<<"Ratio in section 2"<<(float)section2_count/(float)section2_area<<std::endl;
	if((float)section2_count/(float)section2_area > 0.75)
		new_rotation=60;
	else{
		for(size_t i= 0; i < (real_cols/2 - 25)/2; i+=1 ){
			for(size_t j=0;j<max_segments;j++) {
		
			Section section = stixels[i*max_segments+j];
			//fp<<section.type<<","<<section.vB<<","<<section.vT<<","<<section.disparity<<";"<<std::endl;
			if(section.vT<= level_height && section.type == OBJECT)
				section1_count+=5*(section.vT-section.vB);
				else if (section.vB < level_height && section.vT >=level_height && section.type == OBJECT)
					section1_count+=5*(level_height-section.vB);		
			}	
		}
		if((float)section1_count/(float)section2_area > 0.75)
			new_rotation=30;
	}

	for(size_t i=(real_cols/2 + 25) ; i < (real_cols/2 + 25) +((real_cols/2 - 25)/2); i+=1 ){
		for(size_t j=0;j<max_segments;j++) {
		
			Section section = stixels[i*max_segments+j];
			//fp<<section.type<<","<<section.vB<<","<<section.vT<<","<<section.disparity<<";"<<std::endl;
			if(section.vT<= level_height && section.type == OBJECT)
				section4_count+=5*(section.vT-section.vB);
				else if (section.vB < level_height && section.vT >=level_height && section.type == OBJECT)
					section4_count+=5*(level_height-section.vB);		
		}	
	}
	std::cout<<"Ratio in section 4"<<(float)section4_count/(float)section2_area<<std::endl;
	if((float)section4_count/(float)section2_area > 0.75)
		new_rotation-=60;
	else{
		for(size_t i= (real_cols/2 + 25) +((real_cols/2 - 25)/2); i < real_cols; i+=1 ){
			for(size_t j=0;j<max_segments;j++) {
		
			Section section = stixels[i*max_segments+j];
			//fp<<section.type<<","<<section.vB<<","<<section.vT<<","<<section.disparity<<";"<<std::endl;
			if(section.vT<= level_height && section.type == OBJECT)
				section5_count+=5*(section.vT-section.vB);
				else if (section.vB < level_height && section.vT >=level_height && section.type == OBJECT)
					section5_count+=5*(level_height-section.vB);		
			}	
		}
		if((float)section5_count/(float)section2_area > 0.75)
			new_rotation-=30;
	}
	if(!new_rotation){
		buffer[0]='x';
		nBytes=strlen(buffer);
		sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);
	}
	else if (new_rotation==30){
		buffer[0]='d';
		nBytes=strlen(buffer);
		sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);
		new_car_speed=10;
	}
	else if (new_rotation==60){
		buffer[0]='d';
		nBytes=strlen(buffer);
		sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);
		new_car_speed=20;
	}
	else if (new_rotation==-60){
		buffer[0]='a';
		nBytes=strlen(buffer);
		sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);
		new_car_speed=20;
	}
	else if (new_rotation==-30){
		buffer[0]='a';
		nBytes=strlen(buffer);
		sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);
		new_car_speed=10;
	}

	int change=(new_car_speed-old_car_speed)/10;
	std::cout<<"Change"<<change<<std::endl;
	if(change<0){		
		std::cout<<"Control for decrease"<<std::endl;
			for(int k=0;k<2*abs(change);k++){		
				buffer[0] = '<';
				nBytes = strlen(buffer);  //UDP communication to the Pi
				sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);	
			}
				buffer[0]='w';
				nBytes = strlen(buffer);  //UDP communication to the Pi
				sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);	
				
			
		}
	
	else {
		
			std::cout<<"Control to increase" << std::endl;
			//int l=car_speed/10;		
			for(int k=0;k<2*change;k++){		
				buffer[0]= '>';
				nBytes = strlen(buffer);  //UDP communication to the Pi
				sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);	
				//buffer[0]='w';
				//nBytes = strlen(buffer);
				//sendto(clientSocket, buffer, nBytes, 0, (struct sockaddr *)&serverAddr,addr_size);
			}
			buffer[0]= 'w';
			nBytes = strlen(buffer);  //UDP communication to the Pi
			sendto(clientSocket,buffer,nBytes,0,(struct sockaddr *)&serverAddr,addr_size);	
	}
	return new_car_speed;	
}
bool FileExists(const char *fname) {
	struct stat buffer;
	return (stat (fname, &buffer) == 0);
}

int write_image(const char* directory, const char* sub_directory, int* l, const char* file_extension, cv::Mat& im_age,std::vector<int>& v){

	char buffer_file[PATH_MAX];
	sprintf(buffer_file, "%s/%s/%i.%s", directory, sub_directory, *l,file_extension);
	cv::imwrite(buffer_file, im_age, v);
	return 1;
}

int main(int argc, char *argv[]) {

	int mode = atoi(argv[1]);
	if(!argv[1] && argc < 7) {
		std::cerr << "Usage: stixels mode max_disparity p1 p2 images_number(-1 for infinite) dir" << std::endl;
		return -1;
	}
	
	clientSocket = socket(PF_INET, SOCK_DGRAM, 0);

  	/*Configure settings in address struct*/
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(5050);
	serverAddr.sin_addr.s_addr = inet_addr("10.42.0.144");
	memset(serverAddr.sin_zero, '\0', sizeof serverAddr.sin_zero);  

  	/*Initialize size variable to be used later on*/
	addr_size = sizeof serverAddr;
	
	//Creating a ZED Camera object	
	sl::Camera zed;
	sl::Mat zed_image_left, zed_image_right, depth_image_show, depth_image;
	cv::Size size(1024,768);
	
	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
	
	//Set Configuration parameters
	sl::InitParameters init_params;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE; //Use PERFORMANCE depth mode
	init_params.coordinate_units = sl::UNIT_MILLIMETER; // Use millimeter units (for depth measurements)
	init_params.camera_fps= 60;
	init_params.camera_resolution= sl::RESOLUTION_HD720;
	
	//Open the camera
	sl::ERROR_CODE err = zed.open(init_params);
	if(err!= sl::SUCCESS)
		exit(-1);
	
	
	const int max_dis = atoi(argv[2]);
	
	//Arguments for the SGM code
	uint8_t p1, p2;
	p1 = atoi(argv[3]);  	  //Value of p1 is the third argument
	p2 = atoi(argv[4]);	  // Value of p2 is the fourth argument
	int count = atoi(argv[5]);
	const char* directory = argv[6];

	DIR *dp;
	
	char abs_dis_dir[PATH_MAX];
	if(!mode) {
	sprintf(abs_dis_dir, "%s/", directory);	//Disparity directory is fixed here
	dp = opendir(abs_dis_dir);
	if (dp == NULL) {
		std::cerr << "Invalid directory: " << abs_dis_dir << std::endl;		
		exit(EXIT_FAILURE);
	}
	
	}	
	
	
	/* Disparity Parameters */
	const float sigma_disparity_object = 1.0f;
	const float sigma_disparity_ground = 2.0f;
	const float sigma_sky = 0.1f; 		// Should be small compared to sigma_dis

	/* Probabilities */
	const float pout = 0.2f;
	const float pout_sky = 0.4f;
	const float pord = 0.2f;
	const float pgrav = 0.1f;
	const float pblg = 0.04f;

	// Must add 1
	const float pground_given_nexist = 0.36f;
	const float pobject_given_nexist = 0.28f;
	const float psky_given_nexist = 0.36f;

	const float pnexist_dis = 0.0f;
	const float pground = 1.0f/3.0f;
	const float pobject = 1.0f/3.0f;
	const float psky = 1.0f/3.0f;

	/* Camera Paramters */
	int vhor;

	// Virtual parameters
	const float focal = 699.836f;
	const float baseline = 0.12f;
	const float camera_center_y = 381.847f;
	const int column_step = 5;
	const int width_margin = 0;

	float camera_tilt;
	const float sigma_camera_tilt = 0.05f;
	float camera_height;
	const float sigma_camera_height = 0.05f;
	float alpha_ground;

	/* Model Parameters */
	const bool median_step = false;
	const float epsilon = 3.0f;
	const float range_objects_z = 1.0f; // in meters

	/* Parameters to add the sgm implementation to the code */
	std::vector<float> times;

	init_disparity_method(p1, p2);

	bool first_time = true;
	Stixels stixles;
	RoadEstimation road_estimation;
	std::vector<float> times_stixel;
	pixel_t *im;
	
	int l=0;
	char stixel_file[PATH_MAX];	
	while (count--) {
	
	l++;	
			
	if (zed.grab() == sl::SUCCESS) {
		        
        zed.retrieveImage(zed_image_left, sl::VIEW_LEFT);  
        zed.retrieveImage(zed_image_right, sl::VIEW_RIGHT);
	}	
				
	cv::Mat dis_right = slMat2cvMat(zed_image_right), dis_left = slMat2cvMat(zed_image_left);
		
	cv::resize(dis_right, dis_right, size);
	cv::resize(dis_left, dis_left, size);
	
	if(!mode) {
	write_image(directory,"left",&l,"png", dis_left,compression_params);
	write_image(directory,"right",&l,"png", dis_right,compression_params);	
	}		
	// Convert images to grayscale
	if (dis_left.channels()>1) {
		cv::cvtColor(dis_left, dis_left, CV_RGB2GRAY);
	}
	
	if (dis_right.channels()>1) {
		cv::cvtColor(dis_right, dis_right, CV_RGB2GRAY);
	}
	
	if(dis_left.rows != dis_right.rows || dis_left.cols != dis_right.cols) {
		std::cerr << "Both images must have the same dimensions" << std::endl;
		return EXIT_FAILURE;
	}
			
	#if LOG
		std::cout << "processing: " << file_left << std::endl;
	#endif
			
	float elapsed_time_sgm;
	cv::Mat disparity_im = compute_disparity_method(dis_left, dis_right, &elapsed_time_sgm);

	#if LOG
		std::cout << "done" << std::endl;
	#endif
		
	times.push_back(elapsed_time_sgm);
	
	if(!mode)
	write_image(directory,"disparities",&l,"png", disparity_im,compression_params);		
		
	cv::Mat dis = disparity_im;		//To be used for the stixel generation
	
	double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
	std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps to compute the depth from the left and right images" << std::endl;
	

	const int rows = dis.rows;
	const int cols = dis.cols;
		
	std::cout<<rows<<cols<<"\n";
			
	if(rows < max_dis) {
		printf("ERROR: Image height has to be equal or bigger than maximum disparity\n");
		first_time = false;
		continue;
	}

	if(rows >= 1024) {
		printf("ERROR: Maximum image height has to be less than 1024\n");
		first_time = false;
		continue;
	}

	if(first_time) {
				
		stixles.SetDisparityParameters(rows, cols, max_dis, sigma_disparity_object, sigma_disparity_ground, sigma_sky);
		stixles.SetProbabilities(pout, pout_sky, pground_given_nexist, pobject_given_nexist, psky_given_nexist, pnexist_dis, pground, pobject, psky, pord, pgrav, pblg);
		stixles.SetModelParameters(column_step, median_step, epsilon, range_objects_z, width_margin);
		stixles.SetCameraParameters(0.0f, focal, baseline, 0.0f, sigma_camera_tilt, 0.0f, sigma_camera_height, 0.0f);
		stixles.Initialize();
		road_estimation.Initialize(camera_center_y, baseline, focal, rows, cols, max_dis);
		CUDA_CHECK_RETURN(cudaMallocHost((void**)&im, rows*cols*sizeof(pixel_t)));
		first_time = false;
	}
			
	if(dis.depth() == CV_8U) {
		for(int i = 0; i < dis.rows; i++) {
			for(int j = 0; j < dis.cols; j++) {
					const pixel_t d = (float) dis.at<uint8_t>(i, j);
					im[i*dis.cols+j] = d;
			}
		}
	} else {
		for(int i = 0; i < dis.rows; i++) {
			for(int j = 0; j < dis.cols; j++) {
					const pixel_t d = (float) dis.at<uint16_t>(i, j)/256.0f;
					im[i*dis.cols+j] = d;
			}
		}
	}
			
	// Compute some camera parameters
	stixles.SetDisparityImage(im);
			
	const bool ok = road_estimation.Compute(im);
			
	if(!ok) {
		printf("Can't compute road estimation1\n");
		first_time = false;
		continue;
	}
			
	// Get Camera Parameters
	camera_tilt = road_estimation.GetPitch();
	camera_height = road_estimation.GetCameraHeight();
	vhor = road_estimation.GetHorizonPoint();
	alpha_ground = road_estimation.GetSlope();
			
	if(camera_tilt == 0 && camera_height == 0 && vhor == 0 && alpha_ground == 0) {
		printf("Can't compute road estimation2\n");
		first_time = false;
		continue;
	}
			
	std::cout << "Camera Parameters -> Tilt: " << camera_tilt << " Height: " << camera_height << " vHor: " << vhor << " alpha_ground: " << alpha_ground << std::endl;
			
	stixles.SetCameraParameters(vhor, focal, baseline, camera_tilt, sigma_camera_tilt, camera_height, sigma_camera_height, alpha_ground);

	const float elapsed_time_ms = stixles.Compute();
	times_stixel.push_back(elapsed_time_ms);

	Section *stx = stixles.GetStixels();
	int *toproad_row= &vhor;	
	
	if(!mode) {
		sprintf(stixel_file, "%s/%s/%i.%s", directory, "stixels", l, "stixels");
		SaveStixels(stx, stixles.GetRealCols(), stixles.GetMaxSections(), stixel_file);
	}

	new_car_speed=MoveCar(stx, toproad_row, stixles.GetRealCols(), stixles.GetMaxSections(), buffer, old_car_speed);
	old_car_speed=new_car_speed;	
	std::cout<<"Continuing...\n";

	}
	


	if(!first_time) {
		stixles.Finish();
		road_estimation.Finish();
	}

	float mean = 0.0f;
	for(int i = 0; i < times_stixel.size(); i++) {
		mean += times_stixel.at(i);
	}
	mean = mean / times_stixel.size();
	std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
	CUDA_CHECK_RETURN(cudaFreeHost(im));
	
	zed.close();
	return 0;
}

//convert MAT_TYPE to CV_TYPE

cv::Mat slMat2cvMat(sl::Mat& input) {
	
	int cv_type = -1;
	switch (input.getDataType()) {
		case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
		case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
		case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
		case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
		case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
		case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
		case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
		case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
		default: break;
	}

	// cv::Mat data requires a uchar* pointer. Therefore, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	//cv::Mat and sl::Mat will share the same memory pointer
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}


