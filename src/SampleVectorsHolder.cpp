#include "SampleVectorsHolder.h"

SampleVectorsHolder::~SampleVectorsHolder()
{
	delete[] sampleVectors;
}

const float* SampleVectorsHolder::getSampleVectors()
{
	return this->sampleVectors;
}

int SampleVectorsHolder::getSampleVectorCount()
{
	return this->sampleVectorsCount;
}

int SampleVectorsHolder::getSampleVectorRows()
{
	return this->sampleVectorRows;
}

int SampleVectorsHolder::getSampleVectorCols()
{
	return this->sampleVectorCols;
}