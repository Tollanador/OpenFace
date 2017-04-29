///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

// Magnification API includes
// #include "windows.h"
// #include "resource.h"
// #include "strsafe.h"
#include "magnification.h"
// magnification.lib;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
		
		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);

		if (!depth_image.empty())
		{
			// Division needed for visualisation purposes
			cv::imshow("depth", depth_image / 2000.0);
		}

	}
}

// Move the mouse cursor
void MousePosition(int x, int y)
{
	double fScreenWidth = ::GetSystemMetrics(SM_CXSCREEN) - 1;
	double fScreenHeight = ::GetSystemMetrics(SM_CYSCREEN) - 1;
	double fx = x*(65535.0f / fScreenWidth);
	double fy = y*(65535.0f / fScreenHeight);
	INPUT  Input = { 0 };
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
	Input.mi.dx = fx;
	Input.mi.dy = fy;
	::SendInput(1, &Input, sizeof(INPUT));
}

int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> files, depth_directories, output_video_files, out_dummy;
	
	// By default try webcam 0
	int device = 0;

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to world or camera coordinates
	bool u;
	string output_codec;
	LandmarkDetector::get_video_input_output_params(files, depth_directories, out_dummy, output_video_files, u, output_codec, arguments);
	
	// The modules that are being used for tracking
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);	

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	
	det_parameters.track_gaze = true;


	// Gaze tracking, previous and change in absolute gaze direction
	/*
	cv::Point3f pregazeDirection0(0, 0, -1);
	cv::Point3f pregazeDirection1(0, 0, -1);
	cv::Point3f deltagazeDirection0(0, 0, -1);
	cv::Point3f deltagazeDirection1(0, 0, -1);
	int MouseX = 1000;
	int MouseY = 500;
	*/
	int smoothMouseX = 1000;
	int smoothMouseY = 500;
	int smoothing = 1000;
	bool MouseControl = true;
	bool MouseCalibrate = true;
	cv::Point3f mingazeDirection0(0, 0, -1);
	cv::Point3f mingazeDirection1(0, 0, -1);
	cv::Point3f maxgazeDirection0(0, 0, -1);
	cv::Point3f maxgazeDirection1(0, 0, -1);
	cv::Point3f mingazeDirection(0, 0, -1);
	cv::Point3f maxgazeDirection(0, 0, -1);
	cv::Point3f midgazeDirection0(0, 0, -1);
	cv::Point3f midgazeDirection1(0, 0, -1);
	float mingazeDiff = 0;
	float maxgazeDiff = 0;
	
	float ScreenX = (float)GetSystemMetrics(SM_CXSCREEN);
	float ScreenY = (float)GetSystemMetrics(SM_CYSCREEN);
	float magFactor = 1.f;
	float MagScreenX = ScreenX / magFactor;
	float MagScreenY = ScreenY / magFactor;
	int xDlg = 0;
	int yDlg = 0;
	
	int loop_count = 0;
	bool event_happening = false;
	int ScreenWidth = ::GetSystemMetrics(SM_CXSCREEN);
	int ScreenHeight = ::GetSystemMetrics(SM_CYSCREEN);


	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;

		// We might specify multiple video files as arguments
		if(files.size() > 0)
		{
			f_n++;			
		    current_file = files[f_n];
		}
		else
		{
			// If we want to write out from webcam
			f_n = 0;
		}
		
		bool use_depth = !depth_directories.empty();	

		// Do some grabbing
		cv::VideoCapture video_capture;
		if( current_file.size() > 0 )
		{
			if (!boost::filesystem::exists(current_file))
			{
				FATAL_STREAM("File does not exist");
				return 1;
			}

			current_file = boost::filesystem::path(current_file).generic_string();

			INFO_STREAM( "Attempting to read from file: " << current_file );
			video_capture = cv::VideoCapture( current_file );
		}
		else
		{
			INFO_STREAM( "Attempting to capture from device: " << device );
			video_capture = cv::VideoCapture( device );

			// Set the resolution
			//video_capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			//video_capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			//video_capture.set(CV_CAP_PROP_AUTOFOCUS, 1); //# turn the autofocus off

			// Read a first frame often empty in camera
			cv::Mat captured_image;
			video_capture >> captured_image;
		}

		if (!video_capture.isOpened())
		{
			FATAL_STREAM("Failed to open video source");
			return 1;
		}
		else INFO_STREAM( "Device or file opened");

		cv::Mat captured_image;
		video_capture >> captured_image;		

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}		
	
		int frame_count = 0;
		
		// saving the videos
		cv::VideoWriter writerFace;
		if (!output_video_files.empty())
		{
			try
 			{
				writerFace = cv::VideoWriter(output_video_files[f_n], CV_FOURCC(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), 30, captured_image.size(), true);
			}
			catch(cv::Exception e)
			{
				WARN_STREAM( "Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
			}
		}

		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		

			// Reading the images
			cv::Mat_<float> depth_image;
			cv::Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			// Get depth image
			if(use_depth)
			{
				char* dst = new char[100];
				std::stringstream sstream;

				sstream << depth_directories[f_n] << "\\depth%05d.png";
				sprintf(dst, sstream.str().c_str(), frame_count + 1);
				// Reading in 16-bit png image representing depth
				cv::Mat_<short> depth_image_16_bit = cv::imread(string(dst), -1);

				// Convert to a floating point depth image
				if(!depth_image_16_bit.empty())
				{
					depth_image_16_bit.convertTo(depth_image, CV_32F);
				}
				else
				{
					WARN_STREAM( "Can't find depth image" );
				}
			}
			
			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);
			
			// Visualising the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			double detection_certainty = clnf_model.detection_certainty;

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			cv::Point3f gazeDirection(0, 0, -1);
			float gazeDiff;
			/*
			float gazeDirectionX;
			float gazeDirectionY;
			float gazeDirectionZ;
			float gazeDiff;
			*/
			if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
			{
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
				/*
				gazeDirection.x = (gazeDirection0.x + gazeDirection1.x) / 2.f;
				gazeDirection.y = (gazeDirection0.y + gazeDirection1.y) / 2.f;
				gazeDirection.z = (gazeDirection0.z + gazeDirection1.z) / 2.f;
				*/
				gazeDiff = sqrtf(
					(gazeDirection0.y - gazeDirection1.y) * (gazeDirection0.y - gazeDirection1.y) +
					(gazeDirection0.x - gazeDirection1.x) * (gazeDirection0.x - gazeDirection1.x));

				/*
				//deltagazeDirection0 = gazeDirection0 - pregazeDirection0;
				//deltagazeDirection1 = gazeDirection1 - pregazeDirection1;

				//float deltaX = gazeDirection0.x - pregazeDirection0.x;
				//float deltaY = gazeDirection0.y - pregazeDirection0.y;
				float deltaX = (gazeDirection0.x - pregazeDirection0.x + gazeDirection1.x - pregazeDirection1.x) / 2;
				float deltaY = (gazeDirection0.y - pregazeDirection0.y + gazeDirection1.y - pregazeDirection1.y) / 2;
				//float deltaZ = (gazeDirection0.z - pregazeDirection0.z + gazeDirection1.z - pregazeDirection1.z) / 2;
				
				int mouse_mult = 100000;
				float mouse_min = 0.005;

				//deltaX = deltaX;
				//deltaY = 0;

				//if (deltagazeDirection0.x > mouse_min) {
				if (abs(deltaX) > mouse_min) {
					MouseX -= deltaX * mouse_mult;
					if (MouseX < 0) { MouseX = 0; }
					else if (MouseX > 2000) { MouseX = 2000; }
				}
				if (abs(deltaY) > mouse_min) {
					MouseY -= deltaY * mouse_mult;
					if (MouseY < 0) { MouseY = 0; }
					else if (MouseY > 1000) { MouseY = 1000; }
				}
				/*
				MouseY += deltagazeDirection0.y * mouse_mult;
				if (MouseY < 0) { MouseY = 0; }
				else if (MouseY > 1000) { MouseY = 1000; }
				*
				//MouseMove(MouseX, MouseY);
				//MouseMove(100,100);
				int smoothing = 40;
				smoothMouseX = (smoothMouseX * smoothing + MouseX) / (smoothing + 1);
				smoothMouseY = (smoothMouseY * smoothing + MouseY) / (smoothing + 1);
				MousePosition(smoothMouseX, smoothMouseY);

				pregazeDirection0 = gazeDirection0;
				pregazeDirection1 = gazeDirection1;
				*/
				if (MouseControl && !MouseCalibrate) 
				{
					/*
					{
						//INFO_STREAM("gaze0: " << gazeDirection0.x << gazeDirection0.y << gazeDirection0.z);
						//INFO_STREAM("gaze1: " << gazeDirection1.x << gazeDirection1.y << gazeDirection1.z);
						if (gazeDirection0.x < mingazeDirection0.x) { mingazeDirection0.x = gazeDirection0.x; }
						if (gazeDirection0.x > maxgazeDirection0.x) { maxgazeDirection0.x = gazeDirection0.x; }
						if (gazeDirection0.y < mingazeDirection0.y) { mingazeDirection0.y = gazeDirection0.y; }
						if (gazeDirection0.y > maxgazeDirection0.y) { maxgazeDirection0.y = gazeDirection0.y; }
						if (gazeDirection0.z < mingazeDirection0.z) { mingazeDirection0.z = gazeDirection0.z; }
						if (gazeDirection0.z > maxgazeDirection0.z) { maxgazeDirection0.z = gazeDirection0.z; }
						if (gazeDirection1.x < mingazeDirection1.x) { mingazeDirection1.x = gazeDirection1.x; }
						if (gazeDirection1.x > maxgazeDirection1.x) { maxgazeDirection1.x = gazeDirection1.x; }
						if (gazeDirection1.y < mingazeDirection1.y) { mingazeDirection1.y = gazeDirection1.y; }
						if (gazeDirection1.y > maxgazeDirection1.y) { maxgazeDirection1.y = gazeDirection1.y; }
						if (gazeDirection1.z < mingazeDirection1.z) { mingazeDirection1.z = gazeDirection1.z; }
						if (gazeDirection1.z > maxgazeDirection1.z) { maxgazeDirection1.z = gazeDirection1.z; }
					}
					/*if (gazeDirection0.x > mingazeDirection0.x && gazeDirection0.x < maxgazeDirection0.x &&
						gazeDirection0.y > mingazeDirection0.y && gazeDirection0.y < maxgazeDirection0.y &&
						gazeDirection0.z > mingazeDirection0.z && gazeDirection0.z < maxgazeDirection0.z &&
						gazeDirection1.x > mingazeDirection1.x && gazeDirection1.x < maxgazeDirection1.x &&
						gazeDirection0.y > mingazeDirection1.y && gazeDirection1.y < maxgazeDirection1.y &&
						gazeDirection0.z > mingazeDirection1.z && gazeDirection1.z < maxgazeDirection1.z) 
					*{
						//float mx0 = (gazeDirection0.x - mingazeDirection0.x) / (maxgazeDirection0.x - mingazeDirection0.x);
						float mx0 = (maxgazeDirection0.x - gazeDirection0.x) / (maxgazeDirection0.x - mingazeDirection0.x);
						float my0 = (gazeDirection0.y - mingazeDirection0.y) / (maxgazeDirection0.y - mingazeDirection0.y);
						float mx1 = (maxgazeDirection1.x - gazeDirection1.x) / (maxgazeDirection1.x - mingazeDirection1.x);
						float my1 = (gazeDirection1.y - mingazeDirection1.y) / (maxgazeDirection1.y - mingazeDirection1.y);
						// int mx = (mx0 + mx1) * ScreenWidth / 4;
						// int my = (my0 + my1) * ScreenHeight / 4;
						int mx = (mx0 + mx1) * ScreenWidth / 2;
						int my = (my0 + my1) * ScreenHeight / 2;
						//MousePosition(mx, my);
						int smoothing = 10;
						smoothMouseX = (smoothMouseX * smoothing + mx) / (smoothing + 1);
						smoothMouseY = (smoothMouseY * smoothing + my) / (smoothing + 1);
						// smoothMouseX = mx;
						// smoothMouseY = my;
						MousePosition(smoothMouseX, smoothMouseY);
					}
						*/
					// Sleep(100);

					gazeDirection.x = (gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x) / 2.f;
					gazeDirection.y = (gazeDirection0.y + gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y) / 2.f;
					/*
					gazeDiff = sqrtf(
						(gazeDirection0.y - gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y)
						* (gazeDirection0.y - gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y) +
						(gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x)
						* (gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x));
					*/
					float gazeDiffToll = (maxgazeDiff - mingazeDiff) * 0.0001;
					/*
					if (gazeDiff > maxgazeDiff + gazeDiffToll)
					{
						event_happening = true;
						if (loop_count == 10)
							Beep(750, 100);
						else
							Sleep(10);
					}
					else if (gazeDiff < mingazeDiff - gazeDiffToll)
					{
						event_happening = true;
						if (loop_count == 10)
							Beep(850, 100);
						else
							Sleep(10);
					}
					else */
					if (gazeDirection.x > mingazeDirection.x && gazeDirection.x < maxgazeDirection.x &&
						gazeDirection.y > mingazeDirection.y && gazeDirection.y < maxgazeDirection.y)
					{
						//*
						if (gazeDiff > maxgazeDiff + gazeDiffToll)
						{
							event_happening = true;
							if (loop_count == 10)
							{
								Beep(750, 100); 
								if (magFactor == 1.f)
								{
									magFactor = 3.f;
									smoothing /= magFactor;
								}
								else
								{
									magFactor = 1.f;
									smoothing = 1000;
								}
								MagScreenX = ScreenX / magFactor;
								MagScreenY = ScreenY / magFactor;

								POINT p;
								if (GetCursorPos(&p))
								{
									xDlg = (int)((float)p.x - MagScreenX / 2.0);
									yDlg = (int)((float)p.y - MagScreenY / 2.0);

									if (xDlg > ScreenX - MagScreenX)
										xDlg = ScreenX - MagScreenX;
									else if (xDlg < 0)
										xDlg = 0;

									if (yDlg > ScreenY - MagScreenY)
										yDlg = ScreenY - MagScreenY;
									else if (yDlg < 0)
										yDlg = 0;
								}
								else
								{
									xDlg = (int)(ScreenX * (1.0 - (1.0 / magFactor)) / 2.0);
									yDlg = (int)(ScreenY * (1.0 - (1.0 / magFactor)) / 2.0);
								}

								BOOL fSuccess = MagSetFullscreenTransform(magFactor, xDlg, yDlg);
								if (fSuccess)
								{
									// If an input transform for pen and touch is currently applied, update the transform
									// to account for the new magnification.
									BOOL fInputTransformEnabled;
									RECT rcInputTransformSource;
									RECT rcInputTransformDest;

									if (MagGetInputTransform(&fInputTransformEnabled, &rcInputTransformSource, &rcInputTransformDest))
									{
										if (fInputTransformEnabled)
										{
											// SetInputTransform(hwndDlg, fInputTransformEnabled);
										}
									}
								}
							}
							else
								Sleep(10);
						}
						else if (gazeDiff < mingazeDiff - gazeDiffToll)
						{
							event_happening = true;
							if (loop_count == 10)
							{
								Beep(850, 100);

								if (magFactor == 1.f)
								{
									magFactor = 2.f;
									smoothing /= magFactor;
								}
								else
								{
									magFactor = 1.f;
									smoothing = 1000;
								}
								MagScreenX = ScreenX / magFactor;
								MagScreenY = ScreenY / magFactor;

								POINT p;
								if (GetCursorPos(&p))
								{
									xDlg = (int)((float)p.x - MagScreenX / 2.0);
									yDlg = (int)((float)p.y - MagScreenY / 2.0);

									if (xDlg > ScreenX - MagScreenX)
										xDlg = ScreenX - MagScreenX;
									else if (xDlg < 0)
										xDlg = 0;

									if (yDlg > ScreenY - MagScreenY)
										yDlg = ScreenY - MagScreenY;
									else if (yDlg < 0)
										yDlg = 0;
								}
								else
								{
									xDlg = (int)(ScreenX * (1.0 - (1.0 / magFactor)) / 2.0);
									yDlg = (int)(ScreenY * (1.0 - (1.0 / magFactor)) / 2.0);
								}

								BOOL fSuccess = MagSetFullscreenTransform(magFactor, xDlg, yDlg);
								if (fSuccess)
								{
									// If an input transform for pen and touch is currently applied, update the transform
									// to account for the new magnification.
									BOOL fInputTransformEnabled;
									RECT rcInputTransformSource;
									RECT rcInputTransformDest;

									if (MagGetInputTransform(&fInputTransformEnabled, &rcInputTransformSource, &rcInputTransformDest))
									{
										if (fInputTransformEnabled)
										{
											// SetInputTransform(hwndDlg, fInputTransformEnabled);
										}
									}
								}
							}
							else
								Sleep(10);
						}
						//*/
					}
					else
					{
						/*
						POINT p;
						if (GetCursorPos(&p))
						{
							if (smoothMouseX != p.x) smoothMouseX = p.x;
							if (smoothMouseY != p.y) smoothMouseY = p.y;
						}

						int smoothing = 1000;
						int smoothMouseX_old = smoothMouseX;
						int smoothMouseY_old = smoothMouseY;
						/*
						if (gazeDirection0.x < mingazeDirection0.x) { smoothMouseX += (mingazeDirection0.x - gazeDirection0.x) * smoothing; } // smoothing; }
						else if (gazeDirection0.x > maxgazeDirection0.x) { smoothMouseX -= (gazeDirection0.x - maxgazeDirection0.x) * smoothing; } // smoothing; }
						if (gazeDirection0.y < mingazeDirection0.y) { smoothMouseY -= (mingazeDirection0.y - gazeDirection0.y) * smoothing; } // smoothing; }
						else if (gazeDirection0.y > maxgazeDirection0.y) { smoothMouseY += (gazeDirection0.y - maxgazeDirection0.y) * smoothing; } // smoothing; }
						*/

						int loop_start = 10;
						int loop_stop = 30;
						int before_sleep = 30;
						int during_sleep = 30;
						int after_sleep = 80;
						
						event_happening = true;

						if (loop_count > loop_start && loop_count < loop_stop)
						{
							POINT p;
							if (GetCursorPos(&p))
							{
								if (smoothMouseX != p.x) smoothMouseX = p.x;
								if (smoothMouseY != p.y) smoothMouseY = p.y;
							}

							// int smoothing = 1000;
							int smoothMouseX_old = smoothMouseX;
							int smoothMouseY_old = smoothMouseY;

							if (gazeDirection.x < mingazeDirection.x) 
								smoothMouseX += (mingazeDirection.x - gazeDirection.x) * smoothing;
							else if (gazeDirection.x > maxgazeDirection.x) 
								smoothMouseX -= (gazeDirection.x - maxgazeDirection.x) * smoothing;

							if (gazeDirection.y < mingazeDirection.y)
								smoothMouseY -= (mingazeDirection.y - gazeDirection.y) * smoothing;
							else if (gazeDirection.y > maxgazeDirection.y)
								smoothMouseY += (gazeDirection.y - maxgazeDirection.y) * smoothing;

							if (smoothMouseX < xDlg) { smoothMouseX = xDlg + MagScreenX; }
							else if (smoothMouseX > xDlg + MagScreenX) { smoothMouseX = xDlg; }
							if (smoothMouseY < yDlg) { smoothMouseY = yDlg + MagScreenY; }
							else if (smoothMouseY > yDlg + MagScreenY) { smoothMouseY = yDlg; }

							if (smoothMouseX != smoothMouseX_old || smoothMouseY != smoothMouseY_old)
								MousePosition(smoothMouseX, smoothMouseY);

							Sleep(during_sleep);
						}
						else if (loop_count == loop_stop)
						{
							Beep(650, 100);
						}
						else if (loop_count == 2 * loop_stop)
						{
							Beep(950, 100);
						}
						else if (loop_count > loop_stop)
						{
							Sleep(after_sleep);
						}
						else
							Sleep(before_sleep);

						/*
						if (gazeDirection.x < mingazeDirection.x) 
						{
							event_happening = true;
							if (loop_count > loop_start && loop_count < loop_stop)
							{
								smoothMouseX += (mingazeDirection.x - gazeDirection.x) * smoothing;
								Sleep(during_sleep);
							}
							else if (loop_count == loop_stop)
							{
								Beep(650, 100);
							}
							else if (loop_count == 2 * loop_stop)
							{
								Beep(950, 100);
							}
							else if (loop_count > loop_stop)
							{
								Sleep(after_sleep);
							}
							else
								Sleep(before_sleep);
							/*
							if (gazeDiff < mingazeDiff - gazeDiffToll)
							{
								event_happening = true;
								if (loop_count == 10)
								{
									Beep(850, 100);
									Beep(950, 100);
								}
								else
									Sleep(10);
							}
							*
						} // smoothing; }
						else if (gazeDirection.x > maxgazeDirection.x) 
						{
							event_happening = true;
							if (loop_count > loop_start && loop_count < loop_stop)
							{
								smoothMouseX -= (gazeDirection.x - maxgazeDirection.x) * smoothing;
								Sleep(during_sleep);
							}
							else if (loop_count == loop_stop)
							{
								Beep(650, 100);
							}
							else if (loop_count == 2 * loop_stop)
							{
								Beep(950, 100);
							}
							else if (loop_count > loop_stop)
							{
								Sleep(after_sleep);
							}
							else
								Sleep(before_sleep);
							/*
							if (gazeDiff < mingazeDiff - gazeDiffToll)
							{
								event_happening = true;
								if (loop_count == 10)
								{
									Beep(950, 100);
									Beep(850, 100);
								}
								else
									Sleep(10);
							}
							*
						} // smoothing; }
						/*
						if (gazeDirection.y < mingazeDirection.y) 
						{
							event_happening = true;
							if (loop_count > loop_start && loop_count < loop_stop)
							{
								smoothMouseY -= (mingazeDirection.y - gazeDirection.y) * smoothing;
								Sleep(during_sleep);
							}
							else if (loop_count == loop_stop)
							{
								Beep(650, 100);
							}
							else if (loop_count == 2 * loop_stop)
							{
								Beep(950, 100);
							}
							else if (loop_count > loop_stop)
							{
								Sleep(after_sleep);
							}
							else
								Sleep(before_sleep);
						} // smoothing; }
						else if (gazeDirection.y > maxgazeDirection.y) 
						{
							event_happening = true;
							if (loop_count > loop_start && loop_count < loop_stop)
							{
								smoothMouseY += (gazeDirection.y - maxgazeDirection.y) * smoothing;
								Sleep(during_sleep);
							}
							else if (loop_count == loop_stop)
							{
								Beep(650, 100);
							}
							else if (loop_count == 2 * loop_stop)
							{
								Beep(950, 100);
							}
							else if (loop_count > loop_stop)
							{
								Sleep(after_sleep);
							}
							else
								Sleep(before_sleep); 
						} // smoothing; }
						*
						if (smoothMouseX < 0) { smoothMouseX = ScreenWidth; }
						else if (smoothMouseX > ScreenWidth) { smoothMouseX = 0; }
						if (smoothMouseY < 0) { smoothMouseY = ScreenHeight; }
						else if (smoothMouseY > ScreenHeight) { smoothMouseY = 0; }
						*

						if (smoothMouseX < xDlg) { smoothMouseX = xDlg + MagScreenX; }
						else if (smoothMouseX > xDlg + MagScreenX) { smoothMouseX = xDlg; }
						if (smoothMouseY < yDlg) { smoothMouseY = yDlg + MagScreenY; }
						else if (smoothMouseY > yDlg + MagScreenY) { smoothMouseY = yDlg; }

						if (smoothMouseX != smoothMouseX_old || smoothMouseY != smoothMouseY_old)
							MousePosition(smoothMouseX, smoothMouseY);
						//*/
					}
				}
				else if (!MouseControl && MouseCalibrate)
				{
					//INFO_STREAM("gaze0: " << gazeDirection0.x << gazeDirection0.y << gazeDirection0.z);
					//INFO_STREAM("gaze1: " << gazeDirection1.x << gazeDirection1.y << gazeDirection1.z);
					//*
					if (gazeDirection0.x < mingazeDirection0.x) { mingazeDirection0.x = gazeDirection0.x; }
					if (gazeDirection0.x > maxgazeDirection0.x) { maxgazeDirection0.x = gazeDirection0.x; }
					if (gazeDirection0.y < mingazeDirection0.y) { mingazeDirection0.y = gazeDirection0.y; }
					if (gazeDirection0.y > maxgazeDirection0.y) { maxgazeDirection0.y = gazeDirection0.y; }
					if (gazeDirection0.z < mingazeDirection0.z) { mingazeDirection0.z = gazeDirection0.z; }
					if (gazeDirection0.z > maxgazeDirection0.z) { maxgazeDirection0.z = gazeDirection0.z; }
					if (gazeDirection1.x < mingazeDirection1.x) { mingazeDirection1.x = gazeDirection1.x; }
					if (gazeDirection1.x > maxgazeDirection1.x) { maxgazeDirection1.x = gazeDirection1.x; }
					if (gazeDirection1.y < mingazeDirection1.y) { mingazeDirection1.y = gazeDirection1.y; }
					if (gazeDirection1.y > maxgazeDirection1.y) { maxgazeDirection1.y = gazeDirection1.y; }
					if (gazeDirection1.z < mingazeDirection1.z) { mingazeDirection1.z = gazeDirection1.z; }
					if (gazeDirection1.z > maxgazeDirection1.z) { maxgazeDirection1.z = gazeDirection1.z; }

					midgazeDirection0.x = mingazeDirection0.x + (maxgazeDirection0.x - mingazeDirection0.x) / 2;
					midgazeDirection1.x = mingazeDirection1.x + (maxgazeDirection1.x - mingazeDirection1.x) / 2;
					midgazeDirection0.y = mingazeDirection0.y + (maxgazeDirection0.y - mingazeDirection0.y) / 2;
					midgazeDirection1.y = mingazeDirection1.y + (maxgazeDirection1.y - mingazeDirection1.y) / 2;

					gazeDirection.x = (gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x) / 2.f;
					gazeDirection.y = (gazeDirection0.y + gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y) / 2.f;
					/*/
					gazeDiff = sqrtf(
						(gazeDirection0.y - gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y)
						* (gazeDirection0.y - gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y) +
						(gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x)
						* (gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x));
					//*/
					if (gazeDirection.x < mingazeDirection.x) { mingazeDirection.x = gazeDirection.x; }
					if (gazeDirection.x > maxgazeDirection.x) { maxgazeDirection.x = gazeDirection.x; }
					if (gazeDirection.y < mingazeDirection.y) { mingazeDirection.y = gazeDirection.y; }
					if (gazeDirection.y > maxgazeDirection.y) { maxgazeDirection.y = gazeDirection.y; }
					//*/
					if (gazeDiff > maxgazeDiff) { maxgazeDiff = gazeDiff; }
					if (gazeDiff < mingazeDiff) { mingazeDiff = gazeDiff; }
					//*/
					event_happening = true;

					if (loop_count > 100)
					{
						Beep(750, 100);
						Beep(850, 100);
						Beep(750, 100);
						MouseControl = true;
						MouseCalibrate = false;
						INFO_STREAM("Mouse Control ON");
					}
					else
					{
						Sleep(10);
					}
				}
			}

			cv::Mat blank_image;
			blank_image = cv::Mat::zeros(captured_image.rows, captured_image.cols, CV_32F);
			
			// blank_image = captured_image.clone();
			// blank_image = grayscale_image.clone();
			visualise_tracking(blank_image, depth_image, clnf_model, det_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);
			/*
			cv::Mat blank_image_mirror = blank_image.clone();
			for (int i = 0; i < captured_image.rows; i++) 
			{
				for (int j = 0; j < captured_image.cols; j++)
				{

				}
			}
			*/

			// output the tracked video
			if (!output_video_files.empty())
			{
				writerFace << captured_image;
			}


			video_capture >> captured_image;
		
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press == 'r')
			{
				clnf_model.Reset();
			}
			// print the current gaze directions
			if (character_press == 'g')
			{
				INFO_STREAM("gaze0: ( " << gazeDirection0.x
					<< ", " << gazeDirection0.y
					<< ", " << gazeDirection0.z
					<< " )");
				INFO_STREAM("gaze1: ( " << gazeDirection1.x
					<< ", " << gazeDirection1.y
					<< ", " << gazeDirection1.z
					<< " )");
			}
			// print the current gaze x directions
			if (character_press == 'x')
			{
				INFO_STREAM("gaze0x: " << gazeDirection0.x << ", gaze1x: " << gazeDirection1.x);
			}
			// print the current gaze y directions
			if (character_press == 'c')
			{
				INFO_STREAM("gaze0y: " << gazeDirection0.y << ", gaze1y: " << gazeDirection1.y);
			}
			// print the current gaze z directions
			if (character_press == 'v')
			{
				INFO_STREAM("gaze0z: " << gazeDirection0.z << ", gaze1z: " << gazeDirection1.z);
			}
			if (character_press == 's')
			{
				INFO_STREAM("Screen Size: " << ::GetSystemMetrics(SM_CXSCREEN) << ", " << ::GetSystemMetrics(SM_CYSCREEN));
			}
			// restart the mouse
			if (character_press == 'm')
			{
				/*
				MouseX = 1000;
				MouseY = 500;
				smoothMouseX = 1000;
				smoothMouseY = 500;
				*/
				if (MouseControl)
				{
					MouseControl = false;
					MouseCalibrate = true;
					INFO_STREAM("Mouse Control OFF");
					
					mingazeDirection0 = gazeDirection0;
					maxgazeDirection0 = gazeDirection0;
					midgazeDirection0 = gazeDirection0;
					mingazeDirection1 = gazeDirection1;
					maxgazeDirection1 = gazeDirection1;
					midgazeDirection1 = gazeDirection1;
					
					MousePosition(ScreenWidth / 2, ScreenHeight / 2);
					smoothMouseX = ScreenWidth / 2;
					smoothMouseY = ScreenHeight / 2;

					gazeDirection.x = 0; // (gazeDirection0.x + gazeDirection1.x - midgazeDirection0.x - midgazeDirection1.x) / 2.f;
					gazeDirection.y = 0; // (gazeDirection0.y + gazeDirection1.y - midgazeDirection0.y - midgazeDirection1.y) / 2.f;
					// gazeDiff = 0; // sqrtf(
					
					mingazeDirection = gazeDirection;
					maxgazeDirection = gazeDirection;
					midgazeDirection0 = gazeDirection0;
					midgazeDirection1 = gazeDirection1;
					maxgazeDiff = gazeDiff;
					mingazeDiff = gazeDiff;
				}
				else 
				{ 
					MouseControl = true;
					MouseCalibrate = false;
					INFO_STREAM("Mouse Control ON");
					INFO_STREAM("Delta gaze0: " << maxgazeDirection0.x - mingazeDirection0.x
										<< ", " << maxgazeDirection0.y - mingazeDirection0.y
										<< ", " << maxgazeDirection0.z - mingazeDirection0.z );
					INFO_STREAM("Delta gaze1: " << maxgazeDirection1.x - mingazeDirection1.x
										<< ", " << maxgazeDirection1.y - mingazeDirection1.y
										<< ", " << maxgazeDirection1.z - mingazeDirection1.z );
					/*
					INFO_STREAM("Delta gaze1x: " << maxgazeDirection1.x - mingazeDirection1.x);
					INFO_STREAM("Delta gaze1y: " << maxgazeDirection1.y - mingazeDirection1.y);
					INFO_STREAM("Delta gaze1z: " << maxgazeDirection1.z - mingazeDirection1.z);
					*/
				}
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;

			if (event_happening)
			{
				loop_count++;
				event_happening = false;
			}
			else
			{
				if (loop_count > 0)
					loop_count = 0;
				Sleep(100);
			}
			// Sleep(100);
		}
		
		frame_count = 0;

		// Reset the model, for the next video
		clnf_model.Reset();
		
		// break out of the loop if done with all the files (or using a webcam)
		if(f_n == files.size() -1 || files.empty())
		{
			done = true;
		}
	}

	return 0;
}

