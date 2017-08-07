#include "CQParameters.h"

CQParameters::CQParameters()
{}

CQParameters::~CQParameters()
{}

bool CQParameters::Exists(const string& key)
{
	return parameter_set.find(key) != parameter_set.end();
}

void CQParameters::WriteHelpInformation()
{
	cout << "PQ=";
	cout << "NCQ=";
	cout << "CQ=";
	cout << "Search=";

	cout << "********* global parameters *********\n";
	cout << "points_count=\n";
	cout << "dictionaries_count=\n";
	cout << "words_count=\n";
	cout << "space_dimension=\n";
	cout << "points_file=\n";
	cout << "output_file_prefix=\n";
	cout << "max_iter=\n";

	cout << "********** PQ parameters *************\n";
	cout << "distortion_tol=\n";
	cout << "read_partition=\n";
	cout << "partition_file=\n";

	cout << "********** NCQ and CQ parameters **********\n";
	cout << "num_sep=\n";
	cout << "~~~~~~~~~~ initial from outside ~~~~~~~~~~\n";
	cout << "initial_from_outside=\n";
	cout << "dictionary_file=\n";
	cout << "binary_codes_file=\n";
	
	cout << "********** CQ parameters ************\n";
	cout << "mu=\n";

	cout << "********** Search parameters ***********\n";
	cout << "queries_count=\n";
	cout << "groundtruth_length=\n";
	cout << "results_length=\n";
	cout << "queries_file=\n";
	cout << "groundtruth_file=\n";
	
	cout << "trained_dictionary_file=\n";
	cout << "trained_binary_codes_file=\n";
	cout << "output_retrieved_results_file=\n";
}

void CQParameters::LoadFromFile(const string parameter_file)
{
	parameter_set.clear();

	string currentLine;
	ifstream inputStream;
	inputStream.open(parameter_file);
	if (!inputStream.good())
	{
		cout << "unable to open configuration file " + parameter_file << endl;
		throw std::logic_error("unable to open configuration file " + parameter_file);
	}
	while (!inputStream.eof())
	{
		std::getline(inputStream, currentLine);
		if (currentLine.find("help") != string::npos)
		{
			WriteHelpInformation();
			return;
		}
		if (currentLine.length() > 0)
		{
			if ('#' == currentLine[0])	// All lines starting with '#' are skipped as comments.
				continue;
			size_t found = currentLine.find('=');
			if (found == string::npos || found != currentLine.find_last_of('='))
			{
				cout << "Error in parsing data " + currentLine << endl;
				throw std::logic_error("Error in parsing data " + currentLine);
			}
			parameter_set.insert(std::pair<string, string>(currentLine.substr(0, found), currentLine.substr(found + 1)));
		}
	}
	inputStream.close();
}