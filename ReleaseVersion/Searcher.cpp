#include "Searcher.h"


Searcher::Searcher(
	const int points_count,
	const int dictionaries_count,
	const int words_count,
	const int space_dimension,
	const int queries_count,
	const int groundtruth_length,
	const int result_length)
	:points_count_(points_count), 
	dictionaries_count_(dictionaries_count),
	words_count_(words_count),
	space_dimension_(space_dimension),
	queries_count_(queries_count),
	groundtruth_length_(groundtruth_length),
	result_length_(result_length)
{
	if (dictionaries_count <= 0 || words_count <= 0 || space_dimension <= 0 || points_count <= 0 || 
		queries_count_ <= 0 || groundtruth_length_ <= 0 || result_length_ <= 0)
	{
		cout << "Search:: bad input parameters\n";
		throw std::logic_error("Bad input parameters");
	}

	queries_ = new QueryType[space_dimension*queries_count];
	memset(queries_, 0, sizeof(QueryType)*space_dimension*queries_count);
	dictionary_ = new DictionaryType[dictionaries_count*words_count*space_dimension];
	memset(dictionary_, 0, sizeof(DictionaryType)*dictionaries_count*words_count*space_dimension);
	binary_codes_ = new CodeType[dictionaries_count*points_count];
	memset(binary_codes_, 0, sizeof(CodeType)*dictionaries_count*points_count);
	
	groundtruth_ = new PointIdType[queries_count*groundtruth_length];
	memset(groundtruth_, 0, sizeof(PointIdType)*queries_count*groundtruth_length);

	results_.resize(queries_count);
	distance_table_.resize(queries_count, vector<DistanceType>(dictionaries_count*words_count));
}

Searcher::~Searcher()
{
	if (groundtruth_)  delete[] groundtruth_;
	if (binary_codes_) delete[] binary_codes_;
	if (dictionary_)   delete[] dictionary_;
	if (queries_)      delete[] queries_;
}

void Searcher::InitQueries(
	const string queries_file,
	const PointStoreType queries_store_type)
{
	cout << "Reading queries in " + queries_file << endl;
	ReadOneDimensionalPoints<QueryType>(queries_file, queries_store_type, queries_, queries_count_, space_dimension_);
}

void Searcher::InitQueries(const QueryType* queries)
{
	cout << "Reading queries...\n";
	memcpy(queries_, queries, sizeof(QueryType)*queries_count_*space_dimension_);
}

void Searcher::InitGroundtruth(
	const string groundtruth_file,
	const PointStoreType groundtruth_store_type)
{
	cout << "Reading groundtruth in " + groundtruth_file << endl;
	ReadOneDimensionalPoints<PointIdType>(groundtruth_file, groundtruth_store_type, groundtruth_, queries_count_, groundtruth_length_);
}

void Searcher::InitGroundtruth(const PointIdType* groundtruth)
{
	cout << "Reading groundtruth...\n";
	memcpy(groundtruth_, groundtruth, sizeof(PointIdType)*queries_count_*groundtruth_length_);
}

void Searcher::InitDictionary(
	const string dictionary_file,
	const PointStoreType dictionary_store_type)
{
	cout << "Reading dictionaries in " + dictionary_file << endl;
	ReadOneDimensionalPoints<DictionaryType>(dictionary_file, dictionary_store_type, dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void Searcher::InitDictionary(const DictionaryType* dictionary)
{
	cout << "Reading dictionaries...\n";
	memcpy(dictionary_, dictionary, sizeof(DictionaryType)*dictionaries_count_*words_count_*space_dimension_);
}

void Searcher::InitBinaryCodes(
	const string binary_codes_file,
	const PointStoreType binary_codes_store_type)
{
	cout << "Reading binary codes in " + binary_codes_file << endl;
	ReadOneDimensionalPoints<CodeType>(binary_codes_file, binary_codes_store_type, binary_codes_, points_count_, dictionaries_count_);
}

void Searcher::InitBinaryCodes(const CodeType* binary_codes)
{
	cout << "Reading binary codes...\n";
	memcpy(binary_codes_, binary_codes, sizeof(CodeType)*dictionaries_count_*points_count_);
}

void Searcher::SaveNearestNeighborsId(const string output_retrieved_results_file)
{
	cout << "Saving retrieved results in " + output_retrieved_results_file << endl;
	ofstream out_results(output_retrieved_results_file, ios::binary);
	if (!out_results.good()) 
	{
		cout << "Bad output retrieved results file stream : " + output_retrieved_results_file << endl;
		throw std::logic_error("Bad output file stream");
	}
	out_results.write((char*)&queries_count_, sizeof(int));
	out_results.write((char*)&result_length_, sizeof(int));
	for (int query_id = 0; query_id < queries_count_; ++query_id)
	{
		for (int length = 0; length < result_length_; ++length)
		{
			out_results.write(reinterpret_cast<char*>(&(results_[query_id][length].second)), sizeof(PointIdType));
		}
	}
	out_results.close();
}

void Searcher::GetDistanceTable(const int query_id)
{
	QueryType* query = &(queries_[query_id * space_dimension_]);
	DistanceType* distance_table_for_current_query = &(distance_table_[query_id][0]);

	for (int word_id = 0; word_id < dictionaries_count_*words_count_; ++word_id)
	{
		float distance = 0;
		DictionaryType* word = &(dictionary_[word_id*space_dimension_]);
		for (int dimension = 0; dimension < space_dimension_; ++dimension)
		{
			distance += (query[dimension] - word[dimension])*(query[dimension] - word[dimension]);
		}
		distance_table_for_current_query[word_id] = distance;
	}
}

void Searcher::GetNearestNeighborsForEachQuery(const int query_id)
{
	GetDistanceTable(query_id);

	DistanceType* distance_table_for_current_query = &(distance_table_[query_id][0]);
	results_[query_id].resize(result_length_, std::make_pair(FLT_MAX, -1));
	vector<DistanceToQueryType> * result_list = &(results_[query_id]);

	std::make_heap(result_list->begin(), result_list->end());
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		float distance = 0;
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			distance += distance_table_for_current_query[dictionary_id*words_count_ + point_codes[dictionary_id]];
		}
		if (distance < result_list->front().first)
		{
			std::pop_heap(result_list->begin(), result_list->end());
			result_list->pop_back();
			result_list->push_back(std::make_pair(distance, point_id));
			std::push_heap(result_list->begin(), result_list->end());
		}
	}
	std::sort(result_list->begin(), result_list->end());
}

void Searcher::GetNearestNeighbors(const string output_retrieved_results_file)
{
	cout << "Searching (after read queries, dictionary, binary codes)...\n";
	for (int query_id = 0; query_id < queries_count_; ++query_id)
	{
		cout << query_id << endl;
		GetNearestNeighborsForEachQuery(query_id);
	}
	SaveNearestNeighborsId(output_retrieved_results_file);
}

void Searcher::GetRecall(
	const vector<int> & retrieved_lengths_considered, 
	const int n_nearest_groundturths)
{
	if (n_nearest_groundturths > groundtruth_length_)
	{
		cout << "too large number of nearest groundtruth neighbors (" << n_nearest_groundturths << ") considered, should be 1 to " << groundtruth_length_ << endl;
		throw std::logic_error("too large number of nearest groundtruth neighbors considered");
	}
	for (int r_id = 0; r_id < retrieved_lengths_considered.size(); ++r_id)
	{
		float recall = GetRecallAt(groundtruth_, groundtruth_length_, retrieved_lengths_considered[r_id], n_nearest_groundturths);
		cout << "recall@" << retrieved_lengths_considered[r_id] << " (T=" << n_nearest_groundturths << "): " << recall << endl;
	}
}

float Searcher::GetRecallAt(
	const PointIdType* groundtruth, 
	const int groundtruth_length,
	const int retrieved_length_considered, 
	const int n_nearest_groundturths)
{
	if (groundtruth == NULL) {
		cout << "Groundtruth is empty!" << endl;
		return 0;
	}
	float recall = 0;
	for (int query_id = 0; query_id < queries_count_; ++query_id)
	{
		int count = 0;
		for (int index = 0; index < retrieved_length_considered && index < results_.size(); ++index) 
		{
			for (int nearest_id = 0; nearest_id < n_nearest_groundturths; ++nearest_id)
			{
				if (results_[query_id][index].second == groundtruth[query_id*groundtruth_length + nearest_id])
				{
					count++;
				}
			}
		}
		recall += count * 1.0 / n_nearest_groundturths;
	}
	return recall / queries_count_;
}

void Searcher::ReadResults(const string results_file, const int queries_count, const int result_length)
{
	cout << "Reading retrieved results in " + results_file << endl;
	queries_count_ = queries_count;
	result_length_ = result_length;

	ifstream results_stream;
	results_stream.open(results_file.c_str(), ios::binary);
	if (!results_stream.good())
	{
		cout << "Bad results stream: " + results_file << endl;
		throw std::logic_error("Bad input points stream");
	}

	int dim = 0, count = 0;
	results_stream.read((char *)&count, sizeof(int));
	results_stream.read((char *)&dim, sizeof(int));
	if (dim != result_length || count != queries_count)
	{
		cout << "unmatched retrieved results dimension\n";
		throw std::logic_error("unmatched dimension!");
	}
	cout << "Dimension of the vector set:" << dim << endl;
	cout << "Number of the vector set:" << count << endl;
	PointIdType id = 0;
	for (int query_id = 0; query_id < queries_count; ++query_id)
	{
		results_[query_id].resize(result_length);
		for (int length = 0; length < result_length; ++length)
		{
			results_stream.read(reinterpret_cast<char*>(&id), sizeof(PointIdType));
			results_[query_id][length] = std::make_pair(1.0, id);
		}
	}

	results_stream.close();
}