
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <cstdio>
#include <cstdio>
#include <dirent.h>
#include <dirent.h>
#include <fstream>

#define RED "\033[0;31m"         
#define GREEN "\033[0;32m"        
#define Color_Off "\033[0m" 
#define BPurple "\033[1;35m"


using namespace std;
using namespace cv;



static bool writeVocabulary( const string& filename, const Mat& vocabulary )
{
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}

static bool readVocabulary( const string& filename, Mat& vocabulary )
{
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        fs["vocabulary"] >> vocabulary;
       // cout << "done" << endl;
        return true;
    }
    return false;
}


bool isequals(const string& a, const string& b)
{
    unsigned int sz = a.size();
    if (b.size() != sz)
        return false;
    for (unsigned int i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

Mat readImgage(string file_address){
	Mat image;
    image = imread(file_address, CV_LOAD_IMAGE_GRAYSCALE);
    return image;
}
void showImage(Mat image){
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
   	imshow( "Display window", image );                   // Show our image inside it.
   	waitKey(0);   
}
void readImages(string address, vector<Mat> &images, std::vector<string> &labels){
   DIR *dir;
   DIR *inside_dir;
   struct dirent *ent;
   if ((dir = opendir (address.c_str() )) != NULL) {
       /* print all the files and directories within directory */
  		while ((ent = readdir (dir)) != NULL) {
   			 string current_dir_name = ent->d_name;

   			 if (current_dir_name == "." || current_dir_name == "..")
   			 	continue;

   			 string current_dir_address = address  + "/" + current_dir_name ;
   			 inside_dir = opendir ( current_dir_address.c_str() );

			while ((ent = readdir (inside_dir)) != NULL) {
				
   			 	string file_name = ent->d_name;
   			 	if (file_name == "." || file_name == "..")
   			 		continue;
	   			cout << "Reading " << current_dir_name << ", file : " << file_name << "                     \r" << std::flush;
	   			images.push_back(readImgage(current_dir_address + "/" + file_name));
	   			labels.push_back(current_dir_name);
   			}

  		}
  		closedir (dir);
	} else {
  		/* could not open directory */
  		//perror ("");
		} 	
		cout << "\nDone reading images , Count = " << images.size() << endl;
}
