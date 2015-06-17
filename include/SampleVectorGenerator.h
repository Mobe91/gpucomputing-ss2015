#include <vector>
#include "SampleVectorsHolder.h"
#include "CIFARImageLoader.h"

using namespace std;

#pragma once

class SampleVectorGenerator
{
private:
	vector<CIFARImageLoader*> loaders;
	int totalPictureCount;
public:
	SampleVectorGenerator(vector<string> rawCIFARFilePaths);
	~SampleVectorGenerator();

	void generateSampleVectorsFromCIFAR(SampleVectorsHolder** out, const int count = 0);
};