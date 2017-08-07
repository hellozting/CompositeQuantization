#pragma once

#include "DataUtil.h"
#include <algorithm>


class Searcher
{
public:
	/**
	* The constructor function.
	*  @param  points_count        The number of points in the dataset.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*  @param  words_count the     The number of words in each dictionary (K).
	*  @param  space_dimension     The dimension of database vector.
	*  @param  queries_count       The number of queries.
	*  @param groundtruth_length             The length of groundtruth neighbors.
	*  @param  result_length       The number of list length retrived from dataset.
	*/
	Searcher(
		const int points_count,
		const int dictionaries_count,
		const int words_count,
		const int space_dimension,
		const int queries_count,
		const int groundtruth_length,
		const int result_length = 1000);

	/**
	* The deconstructor function.
	*/
	~Searcher();


	/**
	* The initial function for points.
	*  @param  queries_file           The filename with queries in .fvecs format or binary format.
	*  @param  queries_store_type     The type of queries, should be FVEC, IVEC or BINARY.
	*/
	void InitQueries(
		const string queries_file,
		const PointStoreType queries_store_type);

	/**
	* The initial function for points.
	*  @param  queries           A one-dimensional array (of length space_dimension_*queries_count_)
	*                            that the first space_dimension_ data is the first query.
	*/
	void InitQueries(const QueryType* queries);

	/**
	* The initial function for points.
	*  @param  groundtruth_file           The filename with groundtruth in .fvecs format or binary format.
	*  @param  groundtruth_store_type     The type of groundtruth, should be IVEC or BINARY.
	*/
	void InitGroundtruth(
		const string groundtruth_file,
		const PointStoreType groundtruth_store_type);

	/**
	* The initial function for points.
	*  @param  groundtruth           A one-dimensional array (of length queries_count_*groundtruth_length_)
	*                                that the first groundtruth_length_ data is the nearest neighbors of the first query.
	*/
	void InitGroundtruth(const PointIdType* groundtruth);


	/**
	* The initial function for dictionary.
	*  @param  dictionary_file        The filename with dictionary in binary format.
	*  @param  dictionary_store_type  The type of dictionary, should be BINARY.
	*/
	void InitDictionary(
		const string dictionary_file,
		const PointStoreType dictionary_store_type);

	/**
	* The initial function for points.
	*  @param  dictionary           The array that stores the dictionary.
	*/
	void InitDictionary(const DictionaryType* dictionary);

	/**
	* The initial function for dictionary.
	*  @param  binary_codes_file        The filename with binary codes in binary format.
	*  @param  binary_codes_store_type  The type of binary codes, should be BINARY.
	*/
	void InitBinaryCodes(
		const string binary_codes_file,
		const PointStoreType binary_codes_store_type);

	/**
	* The initial function for points.
	*  @param  binary_codes           The array that stores the binary codes.
	*/
	void InitBinaryCodes(const CodeType* binary_codes);


	/**
	* This function read results from outside through results_file, after read the GetRecall function can be called to compute recall.
	*  @param results_file     The filename with results in binary format.
	*  @param quereis_count    The number of queries.
	*  @param result_length    The number of list length retrieved from the dataset.
	*/
	void ReadResults(
		const string results_file,
		const int queries_count,
		const int result_length);


	/**
	* The main function that retrieve the nearest neighbors of queries given the dictionay and binary codes.
	*  @param  output_retrieved_results_file         The filename that will be used to save the retrieval results.
	*/
	void GetNearestNeighbors(const string output_retrieved_results_file);

	/**
	* This function computes the performance in terms of recall@R with R being e.g., 1, 10, 100.
	*  @param retrieved_lengths_considered   The set of R parameters.
	*  @param n_nearest_groundturths         The number of nearest groundtruth neighbors considered
	*/
	void GetRecall(
		const vector<int> & retrieved_lengths_considered, 
		const int n_nearest_groundturths);

	/**
	* This function computes the performance in terms of recall@R with R being e.g., 1, 10, 100.
	*  @param groundtruth                    The two-dimensional arrays with groudtruth nearest neighbors for all the queries.
	*  @param groundtruth_length             The length of groundtruth neighbors.
	*  @param retrieved_length_considered    The specific R parameters.
	*  @param n_nearest_groundturths         The number of nearest groundtruth neighbors considered
	*/
	float GetRecallAt(
		const PointIdType* groundtruth, 
		const int groundtruth_length,
		const int retrieved_length_considered, 
		const int n_nearest_groundturths);
	
private:
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private member functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/**
	* This function disallows the use of compiler-generated copy constructor function
	*/
	Searcher(const Searcher&);
	/**
	* This function disallows the use of compiler-generated copy assignment function
	*/
	Searcher& operator=(const Searcher&);


	/** 
	* This function computes the nearest neighbors for the current query.
	*  @param  query_id   The id of the current query.
	*/
	void GetNearestNeighborsForEachQuery(const int query_id);

	/**
	* This function compuste the distance lookup table for the current query.
	*  @param  query_id   The id of the current query.
	*/
	void GetDistanceTable(const int query_id);

	/**
	* This function saves the retrieval results in the output_file.
	*  @param  output_retrieved_results_file         The filename that will be used to save the retrieval results.
	*/
	void SaveNearestNeighborsId(const string output_retrieved_results_file);


	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private member variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/**
	* The number of points in the dataset.
	*/
	int points_count_;
	/**
	* The number of dictionaries (M).
	*/
	int dictionaries_count_;
	/**
	* The number of words in each dictionary (K).
	*/
	int words_count_;
	/**
	* The dimension of database vector.
	*/
	int space_dimension_;
	/**
	* The number of queries.
	*/
	int queries_count_;
	/**
	* The number of list length of the groundturth nearest neighbors.
	*/
	int groundtruth_length_;
	/**
	* The number of list length retrived from the dataset.
	*/
	int result_length_;


	/**
	* A one-dimensional array (of length space_dimension_*queries_count_)
	* that the first space_dimension_ data is the first query.
	*/
	QueryType* queries_;
	/**
	* A one-dimensional array (of length space_dimension_*words_count_*dictionaries_count_)
	* that the first (second) space_dimension_ data is the first (second) word in the first dictionary.
	*/
	DictionaryType* dictionary_;
	/**
	* A one-dimensional array (of length dictionaries_count_*points_count_)
	* that the frist dictionaries_count_ data is the binary codes for the first point.
	*/
	CodeType* binary_codes_;

	
	/**
	* A one-dimensional array (of length queries_count_*groundtruth_length_)
	* that the first groundtruth_length_ data is the nearest neighbors of the first query.
	*/
	PointIdType* groundtruth_;
	/**
	* A two-dimensional array (of queries_count_*result_length_)
	* that the first (second) result_length_ data is the retrive results for the first (second) query vector.
	*/
	vector<vector<DistanceToQueryType>> results_;
	

	/**
	* Temporary variable: a two-dimensional array (of length queries_count_ * (words_count_*dictionaries_count_))
	* that the first (words_count_*dictionaries_count_) data is the distance from each word to the first query.
	*/
	vector<vector<DistanceType>> distance_table_;
};
