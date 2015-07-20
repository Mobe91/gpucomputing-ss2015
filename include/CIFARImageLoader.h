#include <opencv/cv.h>
#include <fstream>

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

using namespace std;

/*EXAMPLE:

try{
	CIFARImageLoader il ("..\\data_batch_1.bin");

	while (true){
		pair<Mat, char> temp = il.getNextImage();
		if (temp.first.empty() || temp.second < 0) break;

		imshow("pic", temp.first);
		cout << "class: " << temp.second << endl;
		waitKey(2000);
	}
}
catch (exception e){ cout << "Couldn't load CIFAR file" << endl; return 1; }
*/

#pragma once
class CIFARImageLoader
{
private:
	ifstream *file;
	ifstream::pos_type size;

	int pictureCount;

	const int IMG_WIDTH;
	const int IMG_HEIGHT;
	const int IMG_SIZE;
public:
	/*
	filename: name of the CIFAR binary file
	thorws an exception if the file does not exist
	*/
	CIFARImageLoader(string filename);
	~CIFARImageLoader();

	/*
	returns the next image and the class from the cifar-10 binary file file
	*/
	pair<cv::Mat,int> getNextImage();

	void CIFARImageLoader::getNextImage(pair<cv::Mat, int> &out);

	/**
	 * Resets the image loader to position 0
	 */
	void reset();

	int getPictureCount();
};
