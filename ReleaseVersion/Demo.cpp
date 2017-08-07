// CompositeQuantizationTraining.cpp : Defines the entry point for the console application.
//
#define _CRTDBG_MAP_ALLOC
#include "CompositeQuantization.h"
#include "NoConstraintCompositeQuantization.h"
#include "ProductQuantization.h"
#include "Searcher.h"
#include "CQParameters.h"
#include <omp.h>



void ProductQuantizationDemo(CQParameters& param)
{
	ProductQuantization PQ(
		param.Get<int>("points_count"),
		param.Get<int>("dictionaries_count"),
		param.Get<int>("words_count"),
		param.Get<int>("space_dimension"));

	PQ.InitPoints(param.Get<string>("points_file"), FVEC);

	KmeansMethod kmeans_method = Lloyd;
	if (param.Get<int>("kmeans_method") == 101)
		kmeans_method = Closure;

	PQ.Training(
		param.Get<int>("max_iter"),
		param.Get<float>("distortion_tol"),
		kmeans_method,
		param.Get<string>("output_file_prefix"),
		param.Get<int>("read_partition"),
		param.Get<string>("partition_file"));
}

void NoConstraintCompositeQuantizationDemo(CQParameters& param)
{
	NoConstraintCompositeQuantization NCQ(
		param.Get<int>("points_count"),
		param.Get<int>("dictionaries_count"),
		param.Get<int>("words_count"),
		param.Get<int>("space_dimension"),
		param.Get<int>("num_sep"));

	NCQ.InitPoints(param.Get<string>("points_file"), FVEC);

	if (param.Get<int>("initial_from_outside") == 1)
	{
		NCQ.InitDictionary(param.Get<string>("dictionary_file"), BINARY);
		NCQ.InitBinaryCodes(param.Get<string>("binary_codes_file"), BINARY);
		NCQ.Training(
			param.Get<int>("max_iter"),
			param.Get<string>("output_file_prefix"),
			false);
	}
	else
	{
		NCQ.Training(
			param.Get<int>("max_iter"),
			param.Get<string>("output_file_prefix"),
			true);
	}
}

void CompositeQuantizationDemo(CQParameters& param)
{
	CompositeQuantization CQ(
		param.Get<int>("points_count"),
		param.Get<int>("dictionaries_count"),
		param.Get<int>("words_count"),
		param.Get<int>("space_dimension"),
		param.Get<int>("num_sep"));

	CQ.InitPoints(param.Get<string>("points_file"), FVEC);

	if (param.Get<int>("initial_from_outside") == 1)
	{
		CQ.InitDictionary(param.Get<string>("dictionary_file"), BINARY);
		CQ.InitBinaryCodes(param.Get<string>("binary_codes_file"), BINARY);
		CQ.Training(
			param.Get<int>("max_iter"),
			param.Get<float>("mu"),
			param.Get<string>("output_file_prefix"),
			false);
	}
	else
	{
		CQ.Training(
			param.Get<int>("max_iter"),
			param.Get<float>("mu"),
			param.Get<string>("output_file_prefix"),
			true);
	}
}

void SearchDemo(CQParameters& param)
{
	Searcher Search(
		param.Get<int>("points_count"),
		param.Get<int>("dictionaries_count"),
		param.Get<int>("words_count"),
		param.Get<int>("space_dimension"),
		param.Get<int>("queries_count"),
		param.Get<int>("groundtruth_length"),
		param.Get<int>("result_length"));

	Search.InitQueries(param.Get<string>("queries_file"), FVEC);
	Search.InitGroundtruth(param.Get<string>("groundtruth_file"), IVEC);
	Search.InitDictionary(param.Get<string>("trained_dictionary_file"), BINARY);
	Search.InitBinaryCodes(param.Get<string>("trained_binary_codes_file"), BINARY);

	Search.GetNearestNeighbors(param.Get<string>("output_retrieved_results_file"));

	vector<int> R;
	R.push_back(1);
	R.push_back(10);
	R.push_back(100);
	Search.GetRecall(R, 1);
}

int main(int argc, char** argv)
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	omp_set_num_threads(omp_get_num_procs());
	cout << "Set threads: " << omp_get_num_procs() << endl;

	//an example of running different quantization methods on 1MSIFT

	{
		CQParameters param;
		param.LoadFromFile("config.txt");

		if (param.Get<int>("PQ") == 1)
		{
			ProductQuantizationDemo(param);
		}

		if (param.Get<int>("NCQ") == 1)
		{
			NoConstraintCompositeQuantizationDemo(param);
		}

		if (param.Get<int>("CQ") == 1)
		{
			CompositeQuantizationDemo(param);
		}

		if (param.Get<int>("Search") == 1)
		{
			SearchDemo(param);
		}
	}
	
	
	_CrtDumpMemoryLeaks();
	return 0;
}