#include "ClosureCluster.h"

namespace KMC
{
	// Fast assignment based on cluster closures
	void ClosureCluster::AssignmentStep()
	{
		if (m_iNThreads > 0) omp_set_num_threads(m_iNThreads);

		// check array to avoid dup computation
		int ** pClusterCheck = new int * [m_iNThreads];
		for (int i = 0; i < m_iNThreads; i++)
		{
			pClusterCheck[i] = new int [m_iNCluster];
			for (int j = 0; j < m_iNCluster; j++) pClusterCheck[i][j] = -1;
		}

		double WCSSD = 0;
#pragma omp parallel for reduction(+ : WCSSD)
		for (int i = 0; i < m_iDataSize; i++)
		{
			int iThread = omp_get_thread_num();
			FloatType fMinDist = MaxDist;
			for (int j = 0; j < m_iCurrentTreeNum; j++)
			{
				int x = (*m_pCode)[j][i]; // the leaf node of point i in tree j
				for (int k = 0; k < m_pInvertedList[x].size(); k++)
				{
					int y = m_pCenterId[m_pInvertedList[x][k]]; // get the cluster id of the k-th member if list[x]
					if (pClusterCheck[iThread][y] != i)
					{
						pClusterCheck[iThread][y] = i;
						FloatType fDist = ComputeDistance((*m_pCenter)[y], (*m_pData)[i], m_iDataDimension);
						if (fDist < fMinDist)
						{
							fMinDist = fDist;
							m_pCenterId[i] = y;
						}
					}
				}
			}

			WCSSD += fMinDist;
		}
		std::cout << WCSSD / m_iDataSize << std::endl;

		for (int i = 0; i < m_iNThreads; i++) delete [] pClusterCheck[i];
		delete [] pClusterCheck;
	}

	void ClosureCluster::Initialization()
	{
		m_pCenterId = new int [m_iDataSize];
		m_pCenter = new Dataset<CenterType> (m_iDataSize, m_iDataDimension);

		// Partition Data by a Random Projection Tree, you can replace it with other patitioning methods
		PartitionTreeBase * pTree = NewPartitionTree(sPartitionMethod, pParams);
		pTree->PartitionData(m_pData, m_pCenterId, m_iNCluster);
		delete pTree;

		// initialize arrays to store forward and inverted index of multiple random partitions
		m_iNPartitions = m_iDataSize / m_iLeafSize;
		m_pCode = new Dataset<int> (m_iMaxTreeNum, m_iDataSize);
		m_pInvertedList.clear();
		for (int i = 0; i < m_iMaxTreeNum * m_iNPartitions; i++)
		{
			std::vector<int> vec;
			vec.clear();
			m_pInvertedList.push_back(vec);
		}
		m_iCurrentTreeNum = 0;

		GenerateNewTrees();

		if (!m_bDynamicTrees) // if not dynamic, generate all trees
		{
			while (m_iCurrentTreeNum < m_iMaxTreeNum) GenerateNewTrees();
		}
	}

	// generate new trees, the number accords the number of threads
	void ClosureCluster::GenerateNewTrees()
	{
		if (m_iNThreads > 0) omp_set_num_threads(m_iNThreads);
		//int iNextTreeNum = m_iCurrentTreeNum + m_iNThreads;
		int iNextTreeNum = m_iCurrentTreeNum + 1;
		if (iNextTreeNum > m_iMaxTreeNum) iNextTreeNum = m_iMaxTreeNum;

#pragma omp parallel for
		for (int i = m_iCurrentTreeNum; i < iNextTreeNum; i++)
		{
			PartitionTreeBase * pTree = NewPartitionTree(sPartitionMethod, pParams);
			pTree->PartitionData(m_pData, (*m_pCode)[i], m_iNPartitions);
			delete pTree;
			for (int j = 0; j < m_iDataSize; j++)
			{
				(*m_pCode)[i][j] += m_iNPartitions * i;
				m_pInvertedList[(*m_pCode)[i][j]].push_back(j);
			}
		}

		m_iCurrentTreeNum = iNextTreeNum;
#ifdef ConsoleOutput
		std::cout << m_iCurrentTreeNum << " trees have been built" << std::endl;
#endif
	}

	void ClosureCluster::RunClustering()
	{
		FloatType TotalRunTime = 0;
		Initialization();
		FloatType LastWCSSD = UpdateStep();
		FloatType LastDrop = -1;
		FloatType LastElapse = 0;
#ifdef ConsoleOutput
		std::cout << "Iteration 0: WCSSD = " << LastWCSSD << std::endl;
#endif
		for (int it = 1; it <= m_iMaxIteration; it++)
		{
			int LastClock = clock();
			AssignmentStep();
			FloatType WCSSD = UpdateStep();
			int Elapse = clock() - LastClock;
			TotalRunTime += FloatType(Elapse) / 1000;
#ifdef ConsoleOutput
			std::cout << "Iteration " << it << ": WCSSD = " << WCSSD << "\tTime cost = " << TotalRunTime << "s" << std::endl;
#endif
			if (LastWCSSD - WCSSD < m_fEpsilon || TotalRunTime > m_iMaxRunTime) break;

			if (m_bDynamicTrees && m_iCurrentTreeNum < m_iMaxTreeNum) // dynamically generate new trees
			{
				FloatType Drop = LastWCSSD - WCSSD;
				if (LastDrop >= 0)
				{
					// Heuristic Criterion
					if (Drop/Elapse < 0.4 * LastDrop/LastElapse)
					{
						GenerateNewTrees();
						LastDrop = -1; // we do not generate new trees in two consecutive iterations
					}
				}
				else
				{
					LastDrop = Drop;
					LastElapse = Elapse;
				}
			}

			LastWCSSD = WCSSD;
		}
		total_WCSSD = LastWCSSD * m_iDataSize;
	}
}