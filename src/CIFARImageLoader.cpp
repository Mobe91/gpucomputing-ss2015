#include "CIFARImageLoader.h"

//see header file for the documentation

CIFARImageLoader::~CIFARImageLoader()
{
	delete file;
}

CIFARImageLoader::CIFARImageLoader(string filename){
	file = new ifstream (filename, ios::in | ios::binary);

	if (!file->is_open()) throw exception();

	file->seekg(0, ios::end);
	size = file->tellg();
	file->seekg(0, ios::beg);
}

pair<Mat, int> CIFARImageLoader::getNextImage(){
	//end of file?
	if (size - file->tellg() < IMG_SIZE*3 + 1) return make_pair(Mat(),-1);

	char *r, *g, *b;
	r = new char[IMG_SIZE];
	g = new char[IMG_SIZE];
	b = new char[IMG_SIZE];
	Mat img;
	char cat;

	file->read(&cat, 1);		//the first byte is the image class
	file->read(r, IMG_SIZE);
	file->read(g, IMG_SIZE);
	file->read(b, IMG_SIZE);
	vector<Mat> channels;
	cv::Size size(IMG_WIDTH, IMG_HEIGHT);
	channels.push_back(Mat(size, CV_8U, (void*)r));
	channels.push_back(Mat(size, CV_8U, (void*)g));
	channels.push_back(Mat(size, CV_8U, (void*)b));

	merge(channels, img);

	return make_pair(img, cat);
}