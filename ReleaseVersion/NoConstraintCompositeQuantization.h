#pragma once
#include "time.h"
#include "DataUtil.h"
#include "ProductQuantization.h"
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#include <mkl.h>


class NoConstraintCompositeQuantization
{
public:
	/**
	* The constructor function.
	*  @param  points_count        The number of points in the dataset.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*  @param  words_count the     The number of words in each dictionary (K).
	*  @param  space_dimension     The dimension of database vector.
	*  @param  num_sep             The number of partitions of the points to accelerate the computation.
	*/
	NoConstraintCompositeQuantization(
		const int points_count,
		const int dictionaries_count,
		const int words_count,
		const int space_dimension,
		const int num_sep = 20);

	/**
	* The deconstructor function.
	*/
	~NoConstraintCompositeQuantization();


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
	* The initial function for dictionary.
	*  @param  dictionary_file        The filename with dictionary in binary format.
	*  @param  dictionary_sotre_type  The type of dictionary, should be BINARY.
	*/
	void InitDictionary(
		const string dictionary_file,
		const PointStoreType dictionary_sotre_type);

	/**
	* The initial function for points.
	*  @param  dictionary          The array that stores the dictionary.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*  @param  words_count the     The number of words in each dictionary (K).
	*/
	void InitDictionary(
		const DictionaryType* dictionary,
		const int dictionaries_count,
		const int words_count);

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
	*  @param  binary_codes        The array that stores the binary codes.
	*  @param  points_count        The number of points in the dataset.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*/
	void InitBinaryCodes(
		const CodeType* binary_codes,
		const int points_count,
		const int dictionaries_count);


	/**
	* This function returns the trained dictionary.
	*/
	const DictionaryType* GetDictionary();

	/**
	* This function returns the trained binary codes.
	*/
	const CodeType* GetBinaryCodes();


	/**
	* The main function that trains the dictionary and the binary codes.
	*  @param  iters                              The iterations of alternating update the three groups of variables.
	*  @param  output_file_prefix                 The prefix of the output file.
	*  @param  initial                            The flag to indicate whether to initial dictionary and binary codes,
	*                                             false -> already initialed from outside
	*                                             true  -> initial inside using results obtained from PQ.
	*/
	void Training(
		const int iters,
		const string output_file_prefix, 
		const bool initial);
private:
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private member functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/**
	* This function disallows the use of compiler-generated copy constructor function
	*/
	NoConstraintCompositeQuantization(const NoConstraintCompositeQuantization&);
	/**
	* This function disallows the use of compiler-generated copy assignment function
	*/
	NoConstraintCompositeQuantization& operator=(const NoConstraintCompositeQuantization&);


	/**
	* The initialization function for dictionary and binary codes.
	*  @param  output_file_prefix                 The prefix of the output file.
	*/
	void InitDictionaryBinaryCodes(const string output_file_prefix);
	/**
	* This function conducts the update dictionary step.
	*/
	void UpdateDictionary();
	/**
	* This function conducts the update binary codes step.
	*/
	void UpdateBinaryCodes();
	/**
	* This function computes the distortions and constants for each point.
	*/
	void GetDistortions();
	/**
	* This function output dictionary in a binary format.
	*/
	void SaveDictionary(const string output_file_prefix);
	/**
	* This function output binary codes in a binary format.
	*/
	void SaveBinaryCodes(const string output_file_prefix);


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
	* The number of partitions of the points to accelerate the computation.
	*/
	int num_sep_;


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
	* Stores the distortion for each point.
	*/
	float* distortions_;
	/**
	* Stores the distortion for all points.
	*/
	float distortion_;


	/**
	* The BB^T of one-dimensional array (of length dictionaries_count*words_count_*dictionaries_count_*words_count).
	*/
	float* binary_multi_binaryTranspose_;
	/**
	* The BB^T of one-dimensional array (of length dictionaries_count*words_count_*dictionaries_count_*words_count)
	* in binary_muti_binaryTranspose_sep_[i] (i is in [0,num_sep_ - 1])
	* to accelerate the computation.
	*/
	vector<vector<float>> binary_multi_binaryTranspose_sep_;
	/**
	* The XB^T of one-dimensional array (of length dictionaries_count*words_count_*space_dimension_).
	*/
	float* points_multi_binaryTranspose_;
	/**
	* The XB^T of one-dimensional array (of length dictionaries_count*words_count_*space_dimension_)
	* in points_multi_binaryTranspose_sep_[i] (i is in [0,num_sep_ - 1])
	* to accelerate the computation.
	*/
	vector<vector<float>> points_multi_binaryTranspose_sep_;
	

	/**
	* Temporary variable to store the u matrix of the svd result.
	*/
	float* u_matrix;
	/**
	* Temporary variable to store the s vector (eigenvalues) of the svd result.
	*/
	float* s_vector;
	/**
	* Temporary variable to store the vt matrix of the svd result.
	*/
	float* vt_matrix;
	/**
	* Temporary variable to store the work of the sgesvd function.
	*/
	float* superb;
};