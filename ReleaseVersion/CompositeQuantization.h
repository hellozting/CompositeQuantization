
#pragma once

#include "lbfgs.h"
#include "time.h"
#include "DataUtil.h"
#include "ProductQuantization.h"
#include "NoConstraintCompositeQuantization.h"
//#pragma comment(lib,"lbfgs.lib")


class CompositeQuantization{
 public:
	/**
	* The constructor function.
	*  @param  points_count        The number of points in the dataset.
	*  @param  dictionaries_count  The number of dictionaries (M).
	*  @param  words_count the     The number of words in each dictionary (K).
	*  @param  space_dimension     The dimension of database vector.
	*  @param  num_sep             The number of partitions of the points to accelerate the gradient computation (default 20).
	*/
	CompositeQuantization(
		const int points_count,
		const int dictionaries_count, 
		const int words_count, 
		const int space_dimension,
		const int num_sep = 20);

	/**
	* The deconstructor function.
	*/
	~CompositeQuantization();


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
	*  @param  dictionary_store_type  The type of dictionary, should be BINARY.
	*/
	void InitDictionary(
		const string dictionary_file,
		const PointStoreType dictionary_store_type);

	/**
	* The initial function for points.
	*  @param  dictionary           The array that stores the dictionary.
	*  @param  dictionaries_count   The number of dictionaries (M).
	*  @param  words_count the      The number of words in each dictionary (K).
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
	* The main function that trains the dictionary and the binary codes initialized by solving a simple problem.
	*  @param  iters               The iterations of alternating update the three groups of variables.
	*  @param  mu                  The penalty parameter (0.0004 for SIFT, 100 for GIST, 0.00001 for MNIST).
	*  @param  output_file_prefix  The prefix of the output file.
	*  @param  initial                            The flag to indicate whether to initial dictionary and binary codes,
	*                                             false -> already initialed from outside
	*                                             true  -> initial inside using results obtained from PQ.
	*/
	void Training(
		const int iters,
		const double mu,
		const string output_file_prefix,
		const bool initial);

	/**
	 * This function gets the binary codes for points (database vectors as learning vectors are used for training)
	   with the trained dictionary fixed.
	 */
	void GetBinaryCodes(
		const PointType* points,
		const DictionaryType* dictionary,
		CodeType* binary_codes, 
		const int iters);

 private:
	 /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private member functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	 /**
	  * This function disallows the use of compiler-generated copy constructor function
	  */
	 CompositeQuantization(const CompositeQuantization&);
	 /**
	 * This function disallows the use of compiler-generated copy assignment function
	 */
	 CompositeQuantization& operator=(const CompositeQuantization&);


	 /**
	 * This function intialize the lbfgs parameter for usage in LBFGS method.
	 */
	 void InitLbfgsParam();
	 /**
	 * The initialization function for dictionary and binary codes.
	 *  @param  output_file_prefix                 The prefix of the output file.
	 */
	 void InitDictionaryBinaryCodes(const string output_file_prefix);


	 /**
	 * This function conducts the update epsilon step.
	 */
	 void UpdateEpsilon();
	 /**
	 * This function conducts the update dictionary step.
	 */
	 void UpdateDictionary();
	 /**
	 * This function conducts the update binary codes step.
	 */
	 void UpdateBinaryCodes();


	 /**
	 * This function computes the inner products between dictionary words from different dictionaries.
	 *  @param  dictionary    A one-dimensional array (of length space_dimension_*words_count_*dictionaries_count_)
	 *                        that the first (second) space_dimension_ data is the first (second) word in the first dictionary.
	 */
	 void GetDictionaryCrossProducts(const DictionaryType* dictionary);
	 /**
	 * This function computes the inner products between dictionary words from different dictionaries
	   (called by lbfgs evaluate function).
	 *  @param  dictionary    A one-dimensional array (of length space_dimension_*words_count_*dictionaries_count_)
	 *                        that the first (second) space_dimension_ data is the first (second) word in the first dictionary.
	 */
	 void GetDictionaryCrossProducts(const lbfgsfloatval_t* dictionary);
	 /**
	 * This function computes the distortions and constants for each point.
	 */
	 void GetDistortionsConstants();


	 /**
	 * This function output dictionary in a binary format.
	 */
	 void SaveDictionary(const string output_file_prefix);
	 /**
	 * This function output binary codes in a binary format.
	 */
	 void SaveBinaryCodes(const string output_file_prefix);


	 /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ friend functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	 /**
	 * Callback interface to provide objective function and gradient evaluations.
	 * 
	 *  The lbfgs() function call this function to obtain the values of objective
	 *  function and its gradients when needed. A client program must implement
	 *  this function to evaluate the values of the objective function and its
	 *  gradients, given current values of variables.
	 *
	 *  @param  instance        The user data sent for lbfgs() function by the client.
	 *  @param  x               The current values of variables.
	 *  @param  g               The gradient vector. The callback function must compute
	 *                          the gradient values for the current variables.
	 *  @param  n               The number of variables 
	                            (equals the number of entries in dictionary, i.e., space_dimension_*words_count_*dictionaries_count_).
	 *  @param  step            The current step of the line search routine.
	 *  @retval lbfgsfloatval_t The value of the objective function for the current
	 *                          variables.
	 */
	 friend static lbfgsfloatval_t evaluate(
		 void *instance,
		 const lbfgsfloatval_t *x,
		 lbfgsfloatval_t *g,
		 const int n,
		 const lbfgsfloatval_t step
		 );
	 /**
	 * Callback interface to receive the progress of the optimization process.
	 *
	 *  The lbfgs() function call this function for each iteration. Implementing
	 *  this function, a client program can store or display the current progress
	 *  of the optimization process.
	 *
	 *  @param  instance    The user data sent for lbfgs() function by the client.
	 *  @param  x           The current values of variables.
	 *  @param  g           The current gradient values of variables.
	 *  @param  fx          The current value of the objective function.
	 *  @param  xnorm       The Euclidean norm of the variables.
	 *  @param  gnorm       The Euclidean norm of the gradients.
	 *  @param  step        The line-search step used for this iteration.
	 *  @param  n           The number of variables
	                        (equals the number of entries in dictionary, i.e., space_dimension_*words_count_*dictionaries_count_).
	 *  @param  k           The iteration count.
	 *  @param  ls          The number of evaluations called for this iteration.
	 *  @retval int         Zero to continue the optimization process. Returning a
	 *                      non-zero value will cancel the optimization process.
	 */
	 friend static int progress(
		 void *instance,
		 const lbfgsfloatval_t *x,
		 const lbfgsfloatval_t *g,
		 const lbfgsfloatval_t fx,
		 const lbfgsfloatval_t xnorm,
		 const lbfgsfloatval_t gnorm,
		 const lbfgsfloatval_t step,
		 int n,
		 int k,
		 int ls
		 );


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
	* The number of partitions of the points to accelerate the gradient computation (default 20).
	*/
	int num_sep_;
	/**
	* The LBFGS parameter, its property can be found in the document of lib-lbfgs.
	*/
	lbfgs_parameter_t lbfgs_param_;


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
	 * Stores the inter-dictionary-element-product for each point.
	 */
	float* constants_;
	/**
	* Stores the inter-dictionary-element-product for all points.
	*/
	float constant_;


	/**
	 * The introduced constant inter-dictionary-element-product initialized by 0.
	 */
	double epsilon_;
	/**
	 * The penalty parameter selected by validation (equals 4*mu in the paper).
	 *  The value choosed in the experiment is 
	 *    1MSIFT         0.0004
	 *    1MGIST         100
	 *    1BSIFT         0.0004
	 *    MNIST          0.00001
	 */
	double mu_;
	

	/**
	* The temporary variable (of length space_dimension_*words_count_*dictionaries_count_)
	* to accelerate the gradient computation.
	*/
	vector<vector<float>> dictionary_gradient_sep_;
	/**
	* The temporary variable storing the inner products between dictionary words from different dictionaries.
	*/
	vector<vector<float>> dictionary_cross_products_;
};