#pragma once
#include "ClusterCommon.h"
#include "Dataset.h"
#include "Distance.h"
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <ctime>

namespace KMC
{
    class Parameters
    {
    public:
        void Add(const std::string& key, const std::string& val)
        {
            if(m_params.find(key) != m_params.end()) 
			{
                throw KMCException(("duplicate keys in params: " + key).c_str());
			}
            m_params[key] = val;
        }

		void Set(const std::string& key, const std::string& val)
		{
			m_params[key] = val;
		}

        template<class T>
        void Get(const std::string& key, T& val) const
        {
            t_params::const_iterator p = m_params.find(key);
            if(p != m_params.end())
			{
                val = StringToValue<T>(p->second);
			}
            else
			{
                throw KMCException(("the parameter can't be found: " + key).c_str());
			}
        }

        template<class T>
        bool Get(const std::string& key, T& val, const T& defaultValue) const
        {
            t_params::const_iterator p = m_params.find(key);
            if(p != m_params.end())
            {
                val = StringToValue<T>(p->second);
                return true;
            }
            else
            {
                val = defaultValue;
                return false;
            }
        }

		template<class T>
        T Get(const std::string& key, const T& defaultValue) const
        {
			T val;
            t_params::const_iterator p = m_params.find(key);
            if(p != m_params.end())
			{
                val = StringToValue<T>(p->second);
			}
            else
			{
                val = defaultValue;
			}
			return val;
        }

		template<class T>
        T Get(const std::string& key) const
        {
			T val;
            t_params::const_iterator p = m_params.find(key);
            if(p != m_params.end())
			{
                val = StringToValue<T>(p->second);
			}
            else
			{
                throw KMCException(("the parameter can't be found: " + key).c_str());
			}
			return val;
        }
        
        bool Exists(const std::string& key)
        {
            return m_params.find(key) != m_params.end();
        }

        void LoadFromFile(const std::string fileName)
        {
            m_params.clear();

            std::string currentLine;
            std::ifstream inputStream;
            inputStream.open(fileName);

            if(inputStream.is_open() == false)
            {
                std::string message = "unable to open configuration file " + fileName;    
                std::cerr<<std::endl<<message;
                throw std::exception(message.c_str());        
            }

            while(!inputStream.eof())
            {        
                std::getline(inputStream, currentLine);                
                if(currentLine.length() > 0)
                {
				    if('#' == currentLine[0])	// All lines starting with '#' are skipped as comments.
					    continue;

                    std::vector<std::string> tokens = StringSplit(currentLine, "= ");

                    if(tokens.size() == 2)
                    {
                        m_params.insert(std::pair<std::string, std::string>(tokens[0], tokens[1]));  
                        std::cout << tokens[0] << '=' << tokens[1] << std::endl;
                    }
                    else
                    {
                        throw std::exception(("Error in parsing data " + currentLine).c_str());
                    }

                    tokens.clear();
                    tokens.resize(0);
                }
            }    

            inputStream.close();
        }

    private:
        typedef std::map<std::string, std::string> t_params;
        t_params m_params;
    };

	class ClusterBase
	{
	public:
		ClusterBase();
		~ClusterBase();

		virtual void LoadParameters(const Parameters & params) = 0;
		virtual void Initialization() = 0;
		virtual void RunClustering() = 0;

		virtual void SetData(Dataset<DataType> * pData);
		virtual void LoadData(const Parameters & params);
		virtual void OutputResult(const Parameters & params) const;

		virtual const CenterType* GetCenter();
		virtual const int* GetCenterId();

	protected:
		bool m_bOwnData;
		Dataset<DataType> * m_pData;

		int m_iNCluster;
		int m_iDataSize;
		int m_iDataDimension;

		int m_iMaxIteration;
		FloatType m_fEpsilon;

		Dataset<CenterType> * m_pCenter;
		int * m_pCenterId;

		virtual void AssignmentStep() = 0;
		virtual FloatType UpdateStep();

	private:

		ClusterBase(const ClusterBase &);
		ClusterBase & operator = (const ClusterBase &);
	};

	

}