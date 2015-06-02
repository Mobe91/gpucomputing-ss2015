#include <string>
#include "SampleVectorsHolder.h"

using namespace std;

#pragma once

class SampleVectorPersistor
{
public:
	SampleVectorPersistor(string sampleVectorFile);
	~SampleVectorPersistor();

	void persistSampleVectors(SampleVectorsHolder &out);
};