#include "Cluster.h"
#include <algorithm>
#include <string>

namespace KMC
{
	ClusterBase::ClusterBase()
		:m_pData(NULL), m_pCenter(NULL), m_bOwnData(false), m_pCenterId(NULL)
	{
	}

	ClusterBase::~ClusterBase()
	{
		if (m_pData != NULL && m_bOwnData) delete m_pData;
		if (m_pCenter != NULL) delete m_pCenter;
		if (m_pCenterId != NULL) delete [] m_pCenterId;
	}

	void ClusterBase::SetData(Dataset<DataType> * pData)
	{
		if (m_pData != NULL && m_bOwnData) delete m_pData;
		m_pData = pData;
		m_bOwnData = false;
		m_iDataSize = m_pData->R();
		m_iDataDimension = m_pData->C();
	}

	const CenterType* ClusterBase::GetCenter()
	{
		return (*m_pCenter)[0];
	}
	
	const int* ClusterBase::GetCenterId()
	{
		return m_pCenterId;
	}

	// Load in Data set
	void ClusterBase::LoadData(const Parameters & params)
	{
		std::string sDataPath = params.Get<std::string>("DataPath");
		FILE * fp;
		fopen_s(&fp, sDataPath.c_str(), "rb");
        if (m_pData != NULL && m_bOwnData) delete m_pData;
		m_bOwnData = true;
        int R, C;
        fread(&R, sizeof(int), 1, fp);
        fread(&C, sizeof(int), 1, fp);

		int iPartialDataSize, iStartDimension, iEndDimension;

		// Check whether we only need process a subset of the data
		params.Get<int>("PartialDataSize", iPartialDataSize, -1);
		if (iPartialDataSize > 0 && iPartialDataSize < R)
		{
			R = iPartialDataSize;
		}
		params.Get<int>("StartDimension", iStartDimension, 0);
		params.Get<int>("EndDimension", iEndDimension, C);

#ifdef ConsoleOutput
        std::cout << "DataSize = " << R << std::endl << "DataDimension = " << iEndDimension - iStartDimension << std::endl;
#endif
		DataType * pTemp = new DataType [C];
        m_pData = new Dataset<DataType> (R, iEndDimension - iStartDimension);
        for (int i = 0; i < R; i++) 
		{
#ifdef ConsoleOutput
			if ((i+1)*100/R != i*100/R) std::cout << "\rLoading " << (i+1)*100/R << "%" ;
#endif
			fread(pTemp, sizeof(DataType), C, fp);
			for (int j = 0; j < iEndDimension - iStartDimension; j++)
			{
				(*m_pData)[i][j] = pTemp[j+iStartDimension];
			}
		}
		delete [] pTemp;
		std::cout << std::endl;
        fclose(fp);

		m_iDataSize = m_pData->R();
		m_iDataDimension = m_pData->C();

	}

	void ClusterBase::OutputResult(const Parameters & params) const
	{
		std::string sOutputFilename = params.Get<std::string>("OutputPrefix");
		char sCenterFilename[255];
		char sAssignFilename[255];

		FILE * fp;

		// Output the center vectors and cluster id of each data vector to text file
		if (params.Get<int>("OutputTextResult", 0) == 1)
		{
			sprintf(sCenterFilename, "%s.center.txt", sOutputFilename.c_str());
			fp = fopen(sCenterFilename, "w");
			fprintf(fp, "%d %d\n", m_iNCluster, m_iDataDimension);
			for (int i = 0; i < m_iNCluster; i++)
			{
				for (int j = 0; j < m_iDataDimension; j++) fprintf(fp, "%f ", float((*m_pCenter)[i][j]));
				fprintf(fp, "\n");
			}
			fclose(fp);

			sprintf(sAssignFilename, "%s.assign.txt", sOutputFilename.c_str());
			fp = fopen(sAssignFilename, "w");
			fprintf(fp, "%d %d\n", m_iDataSize, m_iNCluster);
			for (int i = 0; i < m_iDataSize; i++) fprintf(fp, "%d\n", m_pCenterId[i]);
			fclose(fp);
		}
		
		// Output the center vectors and cluster id of each data vector to binary file
		if (params.Get<int>("OutputBinaryResult", 0) == 1)
		{
			sprintf(sCenterFilename, "%s.center.bin", sOutputFilename.c_str());
			fp = fopen(sCenterFilename, "wb");
			fwrite(&m_iNCluster, sizeof(int), 1, fp);
			fwrite(&m_iDataDimension, sizeof(int), 1, fp);
			for (int i = 0; i < m_iNCluster; i++)
			{
				fwrite((*m_pData)[i], sizeof(CenterType), m_iDataDimension, fp);
			}
			fclose(fp);

			sprintf(sAssignFilename, "%s.assign.bin", sOutputFilename.c_str());
			fp = fopen(sAssignFilename, "wb");
			fwrite(&m_iDataSize, sizeof(int), 1, fp);
			fwrite(&m_iNCluster, sizeof(int), 1, fp);
			fwrite(m_pCenterId, sizeof(int), m_iDataSize, fp);
			fclose(fp);
		}
	}

	// Update step in the Lloyd Iteration
	// Handle the problem of empty cluster by assigning isolated points to the empty clusters
	FloatType ClusterBase::UpdateStep()
	{
		// Count the size of each cluster
		int * pClusterSize = new int [m_iNCluster];
		memset(pClusterSize, 0, sizeof(int) * m_iNCluster);
		for (int i = 0; i < m_iDataSize; i++) 
		{
			pClusterSize[m_pCenterId[i]]++;
		}

		// Check whether empty cluster exists
		// Check the max and min size of a non-empty cluster
		int iEmptyClusterNum = 0;
		int iMaxClusterSize = 0;
		int iMinClusterSize = m_iDataSize;
		for (int i = 0; i < m_iNCluster; i++)
		{
			if (pClusterSize[i] > 0)
			{
				if (pClusterSize[i] > iMaxClusterSize) iMaxClusterSize = pClusterSize[i];
				if (pClusterSize[i] < iMinClusterSize) iMinClusterSize = pClusterSize[i];
			}
			else iEmptyClusterNum++;
		}

#ifdef ConsoleOutput
		std::cout << "# empty clusters = " << iEmptyClusterNum << "; Max cluster size = " << iMaxClusterSize << "; Min cluster size = " << iMinClusterSize << std::endl;
#endif
		// Handle the problem of empty clusters
		if (iEmptyClusterNum > 0)
		{
#ifdef ConsoleOutput
			std::cout << "Fixing empty clusters... " << std::endl;
#endif

			KeyScorePair * pairs = new KeyScorePair [m_iDataSize];
#pragma omp parallel for
			for (int i = 0; i < m_iDataSize; i++)
			{
				pairs[i].Key = i;
				pairs[i].Score = ComputeDistance((*m_pCenter)[m_pCenterId[i]], (*m_pData)[i], m_iDataDimension);
			}
			std::sort(pairs, pairs + m_iDataSize, KeyScorePair::Compare);
			int k = m_iDataSize - 1;
			for (int i = 0; i < m_iNCluster; i++)
			{
				if (pClusterSize[i] == 0)
				{
					while (pClusterSize[m_pCenterId[pairs[k].Key]] < 2) k--;
					pClusterSize[m_pCenterId[pairs[k].Key]]--;
					m_pCenterId[pairs[k].Key] = i;
					pClusterSize[i] = 1;
				}
			}
			delete [] pairs;
		}

		// Update center vectors
		for (int i = 0; i < m_iNCluster; i++)
		{
			pClusterSize[i] = 0;
			memset((*m_pCenter)[i], 0, sizeof(CenterType)*m_iDataDimension);
		}

		for (int i = 0; i < m_iDataSize; i++)
		{
			pClusterSize[m_pCenterId[i]]++;
			for (int j = 0; j < m_iDataDimension; j++) (*m_pCenter)[m_pCenterId[i]][j] += (*m_pData)[i][j];
		}

		iEmptyClusterNum = 0;
		for (int i = 0; i < m_iNCluster; i++)
		{
			if (pClusterSize[i] > 0)
			{
				for (int j = 0; j < m_iDataDimension; j++) (*m_pCenter)[i][j] /= pClusterSize[i];
			}
			else iEmptyClusterNum++;
		}
		if (iEmptyClusterNum > 0)
		{
			std::cout << "Error: found " << iEmptyClusterNum << " empty clusters after fixing" << std::endl;
			system("pause");
		}
		delete [] pClusterSize;

		// Calculate WCSSD
		double WCSSD = 0.0f;

#pragma omp parallel for reduction(+ : WCSSD)
		for (int i = 0; i < m_iDataSize; i++)
		{
			WCSSD += ComputeDistance((*m_pCenter)[m_pCenterId[i]], (*m_pData)[i], m_iDataDimension);
		}
		return FloatType(WCSSD / m_iDataSize);
	}



}