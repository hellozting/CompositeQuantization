#pragma once
#include "Cluster.h"
#include "PartitioningTree.h"
#include <vector>
#include <omp.h>

namespace KMC
{
	class ClosureCluster: public ClusterBase
	{
	public:
		virtual void LoadParameters(const Parameters & params)
		{
			// Parameters for general K-Means
			params.Get<int>("NCluster", m_iNCluster, 10);
			params.Get<int>("MaxIteration", m_iMaxIteration, 10);
			params.Get<FloatType>("Epsilon", m_fEpsilon, FloatType(1e-3));
			params.Get<std::string>("PartitionMethod", sPartitionMethod, "Rptree");
			params.Get<int>("MaxRunTime", m_iMaxRunTime, 36000);
			params.Get<int>("NThreads", m_iNThreads, -1);
			if (m_iNThreads > 0) 
			{
				omp_set_num_threads(m_iNThreads);
			}
			else 
			{
				m_iNThreads = omp_get_num_threads();
			}

			// Parameters for closure algorithm
			params.Get<int>("Closure_MaxTreeNum", m_iMaxTreeNum, 10);
			params.Get<int>("Closure_LeafSize", m_iLeafSize, 200);
			params.Get<int>("Closure_DynamicTrees", m_bDynamicTrees, 0);
			pParams = params;
		}

		virtual void RunClustering();
		~ClosureCluster()
		{
			if (m_pCode != NULL) delete m_pCode;
		}

		FloatType total_WCSSD;

	protected:
		virtual void Initialization();
		virtual void AssignmentStep();

	private:
		std::string sPartitionMethod;
		Parameters pParams;

		Dataset<int> * m_pCode;
		std::vector<std::vector<int>> m_pInvertedList;

		int m_iMaxTreeNum;
		int m_iCurrentTreeNum;
		int m_iNThreads;
		int m_iNPartitions;
		int m_iLeafSize;

		// 0: generate all trees at first; 1: genearte trees dynamically
		int m_bDynamicTrees;

		int m_iMaxRunTime;

		void GenerateNewTrees();
	};
}