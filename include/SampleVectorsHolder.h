using namespace std;

#pragma once

class SampleVectorsHolder
{
private:
	const float* sampleVectors;
	const int sampleVectorsCount;
	const int sampleVectorRows;
	const int sampleVectorCols;

public:
	SampleVectorsHolder(const float* sampleVectors, const int sampleVectorsCount, const int sampleVectorRows, const int sampleVectorCols)
		: sampleVectors(sampleVectors), sampleVectorsCount(sampleVectorsCount), sampleVectorRows(sampleVectorRows), sampleVectorCols(sampleVectorCols)
	{ }

	~SampleVectorsHolder();

	const float* getSampleVectors();

	int getSampleVectorCount();

	int getSampleVectorRows();

	int getSampleVectorCols();
};