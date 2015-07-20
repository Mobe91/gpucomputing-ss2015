#include "CIFARImageLoader.h"

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

using namespace cv;
//see header file for the documentation

CIFARImageLoader::~CIFARImageLoader()
{
	delete file;
}

CIFARImageLoader::CIFARImageLoader(string filename) : IMG_WIDTH(32), IMG_HEIGHT(32), IMG_SIZE(IMG_WIDTH * IMG_HEIGHT)
{
	file = new ifstream (filename, ios::in | ios::binary);

	if (!file->is_open()) throw exception();

	file->seekg(0, ios::end);
	size = file->tellg();
	file->seekg(0, ios::beg);

	pictureCount = size / (IMG_SIZE * 3 + 1);
}

pair<Mat, int> CIFARImageLoader::getNextImage()
{
	pair<Mat, int> labeledImg;
	this->getNextImage(labeledImg);
	return labeledImg;
}

void CIFARImageLoader::getNextImage(pair<Mat, int> &out)
{
	//end of file?
	ifstream::pos_type remaining = this->size - file->tellg();
	if (remaining < IMG_SIZE * 3 + 1) {
		out.second = -1;
		return;
	}

	char *r, *g, *b;
	r = new char[IMG_SIZE];
	g = new char[IMG_SIZE];
	b = new char[IMG_SIZE];

	out.second = 0;
	file->read((char*) &out.second, 1);		//the first byte is the image class
	file->read(r, IMG_SIZE);
	file->read(g, IMG_SIZE);
	file->read(b, IMG_SIZE);
	vector<Mat> channels;
	cv::Size size(IMG_WIDTH, IMG_HEIGHT);
	channels.push_back(Mat(size, CV_8U, (void*)r));
	channels.push_back(Mat(size, CV_8U, (void*)g));
	channels.push_back(Mat(size, CV_8U, (void*)b));

	merge(channels, out.first);
}

void CIFARImageLoader::reset()
{
	file->seekg(0, ios::beg);
}

int CIFARImageLoader::getPictureCount()
{
	return this->pictureCount;
}