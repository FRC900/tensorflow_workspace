#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string>

//#define DEBUG

static int         g_num_frames = 75;
static std::string g_outputdir  = ".";

std::string Behead(std::string my_string)
{
	const size_t found = my_string.rfind("/");

	return my_string.substr(found + 1);
}


void usage(char *argv[])
{
	std::cout << "usage: " << argv[0] << " [-f frames] [-i files] [-o outdir] filename1 [filename2...]" << std::endl << std::endl;
	std::cout << "-f          frames is the number of frames grabbed from a video" << std::endl;
	std::cout << "-o          change output directory from cwd/images to [option]/images" << std::endl;
}


std::vector<std::string> Arguments(int argc, char *argv[])
{
	std::vector<std::string> vid_names;
	vid_names.push_back("");
	if (argc < 2)
	{
		usage(argv);
	}
	else if (argc == 2)
	{
		vid_names[0] = argv[1];
	}
	else
	{
		for (int i = 0; i < argc; i++)
		{
			if (strncmp(argv[i], "-f", 2) == 0)
			{
				try
				{
					g_num_frames = std::stoi(argv[i + 1]);
					if (g_num_frames < 1)
					{
						std::cout << "Must get at least one frame per file!" << std::endl;
						break;
					}
				}
				catch (...)
				{
					usage(argv);
					break;
				}
				i += 1;
			}
			else if (strncmp(argv[i], "-o", 2) == 0)
			{
				g_outputdir = argv[i + 1];
				i++;
			}
			else if (argv[i] != argv[0])
			{
				for ( ; i < argc; i++)
				{
					if (vid_names[0] == "")
					{
						vid_names[0] = argv[i];
					}
					else
					{
						vid_names.push_back(argv[i]);
					}
				}
			}
		}
	}
	return vid_names;
}


typedef std::pair<double, int> Blur_Entry;
void readVideoFrames(const std::string &vidName, int &frameCounter, std::vector<Blur_Entry> &lblur)
{
	cv::VideoCapture frameVideo(vidName);

	lblur.clear();
	frameCounter = 0;
	if (!frameVideo.isOpened())
	{
		return;
	}

	cv::Mat frame;

	cv::Mat temp;
	cv::Mat tempm;
	cv::Mat gframe;
	cv::Mat variancem;

#ifndef DEBUG_FOO
	// Grab a list of frames which have an identifiable
	// object in them.  For each frame, compute a
	// blur score indicating how clear each frame is
	for (frameCounter = 0; frameVideo.read(frame); frameCounter += 1)
	{
		cvtColor(frame, gframe, CV_BGR2GRAY);
		Laplacian(gframe, temp, CV_8UC1);
		meanStdDev(temp, tempm, variancem);
		const double variance = pow(variancem.at<cv::Scalar>(0, 0)[0], 2);
		lblur.push_back(Blur_Entry(variance, frameCounter));
		std::cerr << ".";
	}
#else
	frameCounter = 1;
	lblur.push_back(Blur_Entry(1,187));
#endif
	sort(lblur.begin(), lblur.end(), std::greater<Blur_Entry>());
	std::cout << std::endl << "Read " << lblur.size() << " valid frames from video of " << frameCounter << " total" << std::endl;
}


int main(int argc, char *argv[])
{
	std::vector<std::string> vid_names = Arguments(argc, argv);
	if (vid_names[0] == "")
	{
		std::cout << "Invalid program syntax!" << std::endl;
		return 0;
	}

	// Create output directories
	if (mkdir(g_outputdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
	{
		if (errno != EEXIST)
		{
			std::cerr << "Could not create " << g_outputdir.c_str() << ":";
			perror("");
			return -1;
		}
	}

	// Iterate through each input video
	for (auto vidName = vid_names.cbegin(); vidName != vid_names.cend(); ++vidName)
	{
		std::cout << *vidName << std::endl;

		cv::Mat frame;
		int frame_counter;

		// Grab an array of frames sorted by how clear they are
		std::vector<Blur_Entry> lblur;
		readVideoFrames(*vidName, frame_counter, lblur);
		if (lblur.empty())
		{
			std::cout << "Capture not open; invalid video " << *vidName << std::endl;
			continue;
		}

		cv::VideoCapture frame_video(*vidName);
		int          frame_count = 0;
		std::vector<bool> frame_used(frame_counter);
		const int    frame_range = lblur.size()/(g_num_frames * 1.2);      // Try to space frames out by this many unused frames
		std::cout << "Creating " << g_num_frames << " images spaced at least " << frame_range << " frames apart" << std::endl;
		for (auto it = lblur.begin(); (frame_count < g_num_frames) && (it != lblur.end()); ++it)
		{
			// Check to see that we haven't used a frame close to this one
			// already - hopefully this will give some variety in the frames
			// which are used
			int  this_frame      = it->second;
			bool frame_too_close = false;
			for (int j = std::max(this_frame - frame_range + 1, 0); !frame_too_close && (j < std::min((int)frame_used.size(), this_frame + frame_range)); j++)
			{
				if (frame_used[j])
				{
					frame_too_close = true;
				}
			}

			if (frame_too_close)
			{
				continue;
			}

			frame_used[this_frame] = true;

			frame_video.set(CV_CAP_PROP_POS_FRAMES, this_frame);
			frame_video >> frame;

			std::stringstream write_name;
			write_name << g_outputdir << "/" + Behead(*vidName) << "_" << std::setw(5) << std::setfill('0') << this_frame;
			write_name << ".png";
			std::vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);
			if (cv::imwrite(write_name.str().c_str(), frame, compression_params) == false)
			{
				std::cout << "Error! Could not write file "<<  write_name.str() << std::endl;
				unlink(write_name.str().c_str());
			}
			frame_count += 1;
		}
	}
	return 0;
}

