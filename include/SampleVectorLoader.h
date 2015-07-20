#include <string>
#include "SampleVectorsHolder.h"

///////////////////////////////////////////
// NOT USED
///////////////////////////////////////////

using namespace std;

#pragma once

class SampleVectorLoader
{
public:
	SampleVectorLoader(string sampleVectorFile);
	~SampleVectorLoader();

	void loadSampleVectors(SampleVectorsHolder** out);
};