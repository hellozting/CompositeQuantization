#include "PartitioningTree.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace KMC
{
	PartitionTreeBase * NewPartitionTree(std::string sTreeName)
	{
		if (sTreeName == "Rptree") return new RptreePartition();
		return NULL;
	}

	PartitionTreeBase * NewPartitionTree(std::string sTreeName, const Parameters & params)
	{
		if (sTreeName == "Rptree") return new RptreePartition(params);
		return NULL;
	}

	void RptreePartition::PartitionData(Dataset<DataType> * m_pData, int * pPartitionId, int nPartition)
	{
		int N = m_pData->R();

		// initialize a random id sequence for partition
		int * pIndex = new int [N];
		for (int i = 0; i < N; i++)
		{
			pIndex[i] = i;
		}

		for (int i = N - 1; i > 0; i--)
		{
			int k = (rand()%10000*rand())%(i+1);
			std::swap(pIndex[k], pIndex[i]);
		}

		int CurrentPartitionId = 0;
		// start recursive partitioning
		PartitionDataByRpTree(m_pData, pIndex, pPartitionId, nPartition, CurrentPartitionId, 0, N);

		delete [] pIndex;
	}

	// Partition data into K divisions by recursively random projection partitioning
	void RptreePartition::PartitionDataByRpTree(Dataset<DataType> * m_pData, int * pIndex, int * pPartitionId, int K, 
												int & CurrentPartitionId, int iStartIndex, int iEndIndex)
	{
		if (K == 1)
		{
			for (int i = iStartIndex; i < iEndIndex; i++) pPartitionId[pIndex[i]] = CurrentPartitionId;
			CurrentPartitionId++;
		}
		else
		{
			// Partition into 2 parts, "iMidIndex" is the first index of the second part
			int iMidIndex = ChooseDivisionRPTree(m_pData, pIndex, iStartIndex, iEndIndex);

			// Calculate the number of partitions, proportional to its size, to be allocated to each part
			int leftK = int(floor(float(iMidIndex-iStartIndex)*K/(iEndIndex-iStartIndex)+0.5));
			int rightK = K-leftK;

			// Handle the extreme case
			if (leftK == 0) { leftK = 1; rightK--; }
			if (rightK == 0) { rightK = 1; leftK--; }

			PartitionDataByRpTree(m_pData, pIndex, pPartitionId, leftK, CurrentPartitionId, iStartIndex, iMidIndex);
			PartitionDataByRpTree(m_pData, pIndex, pPartitionId, rightK, CurrentPartitionId, iMidIndex, iEndIndex);
		}
	}

	int RptreePartition::ChooseDivisionRPTree(Dataset<DataType> * m_pData, int * pIndex, int iStartIndex, int iEndIndex)
	{
		int Dim = m_pData->C();
        FloatType * Mean = new FloatType [Dim];
        memset(Mean, 0, Dim*sizeof(FloatType));

		// Some fixed parameters, work for almost all cases


		// We evaluate the quality of a partition plane by a subset of the data
		int iSampleEndIndex  = std::min(iStartIndex+nMaxSample, iEndIndex);
		int nSample = iSampleEndIndex - iStartIndex;

		// Calculate the mean of each dimension
		for (int j = iStartIndex; j < iSampleEndIndex; j++)
		{
			DataType * v = (*m_pData)[pIndex[j]];
			for (int k = 0; k < Dim; k++) Mean[k] += v[k];
		}

		for (int k = 0; k < Dim; k++) Mean[k] /= nSample;
         
        std::vector<KeyScorePair> Variance;
        Variance.clear();
        for (int j = 0; j < Dim; j++)
        {
            Variance.push_back(KeyScorePair(j, 0));
        }

		// Calculate the variance of each dimension
		for (int j = iStartIndex; j < iSampleEndIndex; j++)
		{
            DataType * v = (*m_pData)[pIndex[j]];
			for (int k = 0; k < Dim; k++)
			{
                FloatType dist = v[k] - Mean[k];
				Variance[k].Score += dist*dist;
			}
		}

		// Sort the axis by their variance and pick out the top "nAxis" ones
        std::sort(Variance.begin(), Variance.end(), KeyScorePair::Compare);
        int * AxisIndex = new int [nAxis];
        float * Weight = new float [nAxis];
        float * BestWeight = new float [nAxis];
        float BestVariance = Variance[Dim-1].Score;
        for (int i = 0; i < nAxis; i++)
        {
            AxisIndex[i] = Variance[Dim-1-i].Key;
            BestWeight[i] = 0;
        }

		// Initial best partition plane is set to be the plane perpendicular to the axis with the max variance
        BestWeight[0] = 1;
        float BestMean = Mean[AxisIndex[0]];

        float * Val = new float [nSample];
		// Generate random weights to combine top "nAxis" axis to find better partition plane
        for (int i = 0; i < nIteration; i++)
        {
			// Generate random plane
            float sumweight = 0;
            for (int j = 0; j < nAxis; j++) 
            {
                Weight[j] = float(rand()%10000)/5000.0f - 1.0f;
                sumweight += Weight[j] * Weight[j];
            }
            sumweight = sqrt(sumweight);
            for (int j = 0; j < nAxis; j++) Weight[j] /= sumweight;

			// Calculate the mean of the projection
            float mean = 0;
            for (int j = 0; j < nSample; j++)
            {
                Val[j] = 0;
                for (int k = 0; k < nAxis; k++) Val[j] += Weight[k] * (*m_pData)[pIndex[iStartIndex+j]][AxisIndex[k]];
                mean += Val[j];
            }
            mean /= nSample;

			// Calculate the variance of the projection
            float var = 0;
            for (int j = 0; j < nSample; j++)
            {
                float dist = Val[j] - mean;
                var += dist * dist;
            }

            if (var > BestVariance)
            {
                BestVariance = var;
                BestMean = mean;
                for (int j = 0; j < nAxis; j++) BestWeight[j] = Weight[j];
            }
        }

        delete [] Mean;

        int iLeft = iStartIndex;
		int iRight = iEndIndex-1;

		// decide which child one point belongs
		while (iLeft <= iRight)
		{
            float val = 0;
            for (int k = 0; k < nAxis; k++) val += BestWeight[k] * (*m_pData)[pIndex[iLeft]][AxisIndex[k]];
            if (val < BestMean)
			{
				iLeft++;
			}
			else 
			{
			    std::swap(pIndex[iLeft], pIndex[iRight]);
				iRight--;
			}
		}

		// if all the points in the node are equal,equally split the node into 2 evenly
		if ((iLeft==iStartIndex) || (iLeft==iEndIndex))
		{
			iLeft = (iStartIndex + iEndIndex)/2;
		}

        delete [] Val;
        delete [] AxisIndex;
        delete [] Weight;
        delete [] BestWeight;

		return iLeft;
	}
}