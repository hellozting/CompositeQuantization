#pragma once

#include "Cluster.h"

namespace KMC
{
	// Base class
	class PartitionTreeBase
	{
	public:
		virtual void PartitionData(Dataset<DataType> * m_pData, int * pPartitionId, int nPartition) = 0;		
	};

	// Use Rptree to partition data
	class RptreePartition: public PartitionTreeBase
	{
	public:
		virtual void PartitionData(Dataset<DataType> * m_pData, int * pPartitionId, int nPartition);

		RptreePartition() : nMaxSample(1000), nIteration(100), nAxis(5)
		{
		}

		RptreePartition(const Parameters & params)
		{
			params.Get<int>("Rptree_nMaxSample", nMaxSample, 1000);
			params.Get<int>("Rptree_nIteration", nIteration, 100);
			params.Get<int>("Rptree_nAxis", nAxis, 5);
			//std::cout << nMaxSample << ' ' << nIteration << ' ' << nAxis << std::endl;
		}

	private:
		void PartitionDataByRpTree(Dataset<DataType> * m_pData, int * pIndex, int * pPartitionId, int K, int & CurrentPartitionId, int iStartIndex, int iEndIndex);
		int ChooseDivisionRPTree(Dataset<DataType> * m_pData, int * pIndex, int iStartIndex, int iEndIndex);


		int nMaxSample;
        int nIteration;
        int nAxis;
	};

	// New a partition tree pointer according to its name
	PartitionTreeBase * NewPartitionTree(std::string sTreeName);
	PartitionTreeBase * NewPartitionTree(std::string sTreeName, const Parameters & params);
}