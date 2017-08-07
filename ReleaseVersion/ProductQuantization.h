#pragma once

#include "DataUtil.h"
#include "Kmeans.h"
#include "time.h"
#include "ClosureCluster.h"
#include "Cluster.h"
#include "ClusterCommon.h"
#include <algorithm>

using namespace KMC;

class ProductQuantization
{
public:
	/**
	* The constructor function.
	*  @param  points_count        The number of points in the dataset.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*  @param  words_count         The number of words in each dictionary (K).
	*  @param  space_dimension     The dimension of database vector.
	*/
	ProductQuantization(
		const int points_count,
		const int dictionaries_count,
		const int words_count,
		const int space_dimension);

	/**
	* The deconstructor function.
	*/
	~ProductQuantization();


	/**
	* The initial function for points.
	*  @param  points_file          The filename with points in .fvecs format or binary format.
	*  @param  point_store_type     The type of points, should be FVEC, IVEC or BINARY.
	*/
	void InitPoints(
		const string points_file,
		const PointStoreType point_store_type);

	/**
	* The initial function for points.
	*  @param  points              The array that stores the points.
	*  @param  points_count        The number of points in the dataset.
	*  @param  space_dimension     The dimension of database vector.
	*/
	void InitPoints(
		PointType* points,
		const int points_count,
		const int space_dimension);

	/**
	* This function returns the trained dictionary.
	*/
	const DictionaryType* GetDictionary();

	/**
	* This function returns the trained binary codes.
	*/
	const CodeType* GetBinaryCodes();


	/**
	* The main function that performs product quantization.
	*  @param  max_iters                The maximum iteration of the algorithm.
	*  @param  distortion_tol           The parameter to test the distortion relative variation in consecutive iterations.
	*  @param  kmeans_method            The method of kmeans clustering adopted
	*  @param  output_file_prefix       The prefix of the output file.
	*  @param  read_partition           The flag that indicates whether to read partition outside.
	*  @param  partition_file           The filename with partition in binary format.
	*/
	void Training(
		const int max_iters,
		const double distortion_tol,
		const KmeansMethod kmeans_method,
		const string output_file_prefix,
		const bool read_partition,
		const string partition_file = "");
private:
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private member functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/**
	* This function disallows the use of compiler-generated copy constructor function
	*/
	ProductQuantization(const ProductQuantization&);
	/**
	* This function disallows the use of compiler-generated copy assignment function
	*/
	ProductQuantization& operator=(const ProductQuantization&);


	/**
	* This function output dictionary in a binary format.
	*/
	void SaveDictionary(const string output_file_prefix);
	/**
	* This function output binary codes in a binary format.
	*/
	void SaveBinaryCodes(const string output_file_prefix);
	/**
	* This function output partition in a binary format.
	*/
	void SavePartition(const string output_file_prefix);
	/**
	* This function read partition in a binary format.
	*/
	void ReadPartition(const string partition_file);
	/**
	* This function initial partition in a natural order (for SIFT).
	*/
	void IniNaturalPartition();
	/**
	* This function initial partition in a structure order (for GIST).
	*/
	void IniStructurePartition();


	/**
	* This function performs product quantization training using Lloyd kmeans algorithm.
	*  @param  max_iters                The maximum iteration of the algorithm.
	*  @param  distortion_tol           The parameter to test the distortion relative variation in consecutive iterations.
	*/
	void LloydTraining(const int max_iters, const double distortion_tol);
	/**
	* This function performs product quantization training using Closure cluster algorithm (fast kmeans).
	*  @param  max_iters                The maximum iteration of the algorithm.
	*  @param  distortion_tol           The parameter to test the distortion relative variation in consecutive iterations. 
	*/
	void ClosureTraining(const int max_iters, const double distortion_tol);

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
	* The dimension of subspace.
	*/
	int subspace_dimension_;


	/**
	* A two-dimensional array (of length dictionaries_count_*subspace_dimension_)
	* that partition_[0] containes the indexes of subspace_dimension_ that are divided into the 0th partition.
	*/
	vector<vector<int>> partition_;
	/**
	* A one-dimensional array (of length space_dimension_*points_count_)
	* that the first space_dimension_ data is the first point.
	*/
	PointType* points_;
	/**
	* A flag to indicate whether to manage the points memory.
	*/
	bool own_points_memory_;
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
	* Stores the distortion for all points.
	*/
	float distortion_;
};