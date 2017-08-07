#include "ProductQuantization.h"

ProductQuantization::ProductQuantization(
	const int points_count,
	const int dictionaries_count,
	const int words_count,
	const int space_dimension)
	:points_count_(points_count),
	dictionaries_count_(dictionaries_count),
	words_count_(words_count),
	space_dimension_(space_dimension)
{
	if (points_count <= 0 || dictionaries_count <= 0 || words_count <= 0 || space_dimension <= 0 || space_dimension % dictionaries_count != 0)
	{
		cout << "PQ: bad input parameters\n";
		throw std::logic_error("Bad input parameters");
	}
	subspace_dimension_ = space_dimension / dictionaries_count;
	partition_.resize(dictionaries_count, vector<int>(subspace_dimension_));

	points_ = NULL;
	own_points_memory_ = false;
	dictionary_ = new DictionaryType[dictionaries_count*words_count*space_dimension];
	memset(dictionary_, 0, sizeof(DictionaryType)*dictionaries_count*words_count*space_dimension);
	binary_codes_ = new CodeType[dictionaries_count*points_count];
	memset(binary_codes_, 0, sizeof(CodeType)*dictionaries_count*points_count);

	distortion_ = 0;
}

ProductQuantization::~ProductQuantization()
{
	if (binary_codes_)                    delete[] binary_codes_;
	if (dictionary_)                      delete[] dictionary_;
	if (own_points_memory_ && points_)    delete[] points_;
}

void ProductQuantization::InitPoints(
	const string points_file,
	const PointStoreType point_sotre_type)
{
	cout << "Reading points...\n";
	if (!own_points_memory_)
	{
		points_ = new PointType[space_dimension_*points_count_];
		own_points_memory_ = true;
	}
	ReadOneDimensionalPoints<PointType>(points_file, point_sotre_type, points_, points_count_, space_dimension_);
}

void ProductQuantization::InitPoints(
	PointType* points,
	const int points_count, 
	const int space_dimension)
{
	if (points_count != points_count_ || space_dimension != space_dimension_)
	{
		cout << "unmatched points dimension\n";
		throw std::logic_error("unmatched points dimension");
	}
	cout << "Reading points...\n";
	if (own_points_memory_)
		memcpy(points_, points, sizeof(PointType)*points_count_*space_dimension_);
	else
		points_ = points;
}

const DictionaryType* ProductQuantization::GetDictionary()
{
	return dictionary_;
}

const CodeType* ProductQuantization::GetBinaryCodes()
{
	return binary_codes_;
}

void ProductQuantization::SaveDictionary(const string output_file_prefix)
{
	cout << "Saving dictionary in " + output_file_prefix + "D\n";
	SaveOneDimensionalPoints<DictionaryType>(output_file_prefix + "D", dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void ProductQuantization::SaveBinaryCodes(const string output_file_prefix)
{
	cout << "Saving binary codes in " + output_file_prefix + "B\n";
	SaveOneDimensionalPoints<CodeType>(output_file_prefix + "B", binary_codes_, points_count_, dictionaries_count_);
}

void ProductQuantization::SavePartition(const string output_file_prefix)
{
	cout << "Saving partition in " + output_file_prefix + "partition\n";
	ofstream partition_stream;
	string partition_file = output_file_prefix + "partition";
	partition_stream.open(partition_file.c_str(), ios::binary);
	if (!partition_stream.good()) 
	{
		cout << "Bad output points stream : " + output_file_prefix + "partition\n";
		throw std::logic_error("Bad output partition stream");
	}
	partition_stream.write((char *)&dictionaries_count_, sizeof(int));
	partition_stream.write((char *)&subspace_dimension_, sizeof(int));
	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		partition_stream.write(reinterpret_cast<char*>(&(partition_[dictionary_id][0])), sizeof(int)*subspace_dimension_);
	partition_stream.close();
}

void ProductQuantization::ReadPartition(const string partition_file)
{
	cout << "Reading partition in " + partition_file;
	ifstream partition_stream;
	partition_stream.open(partition_file.c_str(), ios::binary);
	if (!partition_stream.good())
	{
		cout << "Bad input partition stream : " + partition_file << endl;
		throw std::logic_error("Bad input partition stream");
	}
	int count = 0, dim = 0;
	partition_stream.read((char *)&count, sizeof(int));
	partition_stream.read((char *)&dim, sizeof(int));
	if (count != dictionaries_count_ || dim != subspace_dimension_)
	{
		cout << "unmatched partition dimension\n";
		throw std::logic_error("unmatched dimension!");
	}
	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		partition_stream.read(reinterpret_cast<char*>(&(partition_[dictionary_id][0])), sizeof(int)*subspace_dimension_);
	partition_stream.close();
}

void ProductQuantization::IniNaturalPartition()
{
	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
	{
		for (int dim = 0; dim < subspace_dimension_; ++dim)
			partition_[dictionary_id][dim] = dictionary_id*subspace_dimension_ + dim;
	}
}

void ProductQuantization::IniStructurePartition()
{
	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
	{
		for (int dim = 0; dim < subspace_dimension_; ++dim)
			partition_[dictionary_id][dim] = dictionary_id + dim*dictionaries_count_;
	}
}

void ProductQuantization::Training(
	const int max_iters,
	const double distortion_tol,
	const KmeansMethod kmeans_method,
	const string output_file_prefix,
	const bool read_partition,
	const string partition_file)
{
	cout << "Product Quantization Training...\n";
	if (read_partition)
		ReadPartition(partition_file);
	else
		IniNaturalPartition();
	switch (kmeans_method)
	{
	case Lloyd:
		LloydTraining(max_iters, distortion_tol);
		break;
	case Closure:
		ClosureTraining(max_iters, distortion_tol);
		break;
	}
	
	SaveDictionary(output_file_prefix);
	SaveBinaryCodes(output_file_prefix);
	SavePartition(output_file_prefix);

	cout << "Total distortion = " << distortion_ << endl;
}

void ProductQuantization::LloydTraining(
	const int max_iters,
	const double distortion_tol)
{
	distortion_ = 0;
	Kmeans* kmeans = Kmeans_New(points_count_, words_count_, subspace_dimension_, NULL);

	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
	{
		vector<float> subpoints(subspace_dimension_*points_count_);
#pragma omp parallel for
		for (int point_id = 0; point_id < points_count_; ++point_id)
		{
			for (int dim = 0; dim < subspace_dimension_; ++dim)
			{
				subpoints[point_id*subspace_dimension_ + dim] = points_[point_id*space_dimension_ + partition_[dictionary_id][dim]];
			}
		}

		Kmeans_Reset(kmeans, points_count_, words_count_, subspace_dimension_, &subpoints[0]);
		Kmeans_Initialize(kmeans, KmeansInitial_KmeansPlusPlus);
		Kmeans_LloydQuantization(kmeans, max_iters, distortion_tol);

		DictionaryType* current_dictionary = &dictionary_[dictionary_id*words_count_*space_dimension_];
		for (int word_id = 0; word_id < words_count_; ++word_id)
		{
			for (int dim = 0; dim < subspace_dimension_; ++dim)
			{
				current_dictionary[word_id*space_dimension_ + partition_[dictionary_id][dim]] = kmeans->centers_[word_id*subspace_dimension_ + dim];
			}
		}
		for (int point_id = 0; point_id < points_count_; ++point_id)
		{
			binary_codes_[point_id*dictionaries_count_ + dictionary_id] = kmeans->assignments_[point_id];
		}
		distortion_ += kmeans->distortion_;
	}
	Kmeans_Delete(kmeans);
}

void ProductQuantization::ClosureTraining(
	const int max_iters,
	const double distortion_tol)
{
	distortion_ = 0;

	Parameters params;
	params.Set("NCluster", std::to_string(words_count_));
	params.Set("MaxIteration", std::to_string(max_iters));
	params.Set("PartitionMethod", "Rptree");
	params.Set("Rptree_nMaxSample", std::to_string(1000));
	params.Set("Rptree_nIteration", std::to_string(100));
	params.Set("Rptree_nAxis", std::to_string(5));

	params.Set("NThreads", std::to_string(omp_get_num_procs()));

	params.Set("Closure_MaxTreeNum", std::to_string(10));
	params.Set("Closure_LeafSize", std::to_string(int(words_count_ / 10)));
	params.Set("Closure_DynamicTrees", std::to_string(1));

	for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
	{
		vector<float> subpoints(subspace_dimension_*points_count_);
#pragma omp parallel for
		for (int point_id = 0; point_id < points_count_; ++point_id)
		{
			for (int dim = 0; dim < subspace_dimension_; ++dim)
			{
				subpoints[point_id*subspace_dimension_ + dim] = points_[point_id*space_dimension_ + partition_[dictionary_id][dim]];
			}
		}
		Dataset<float> subpoints_(points_count_, subspace_dimension_, &subpoints[0]);

		ClosureCluster CC;
		CC.SetData(&subpoints_);
		CC.LoadParameters(params);
		CC.RunClustering();

		DictionaryType* current_dictionary = &dictionary_[dictionary_id*words_count_*space_dimension_];
		for (int word_id = 0; word_id < words_count_; ++word_id)
		{
			for (int dim = 0; dim < subspace_dimension_; ++dim)
			{
				current_dictionary[word_id*space_dimension_ + partition_[dictionary_id][dim]] = (CC.GetCenter())[word_id*subspace_dimension_ + dim];
			}
		}
		for (int point_id = 0; point_id < points_count_; ++point_id)
		{
			binary_codes_[point_id*dictionaries_count_ + dictionary_id] = (CC.GetCenterId())[point_id];
		}
		distortion_ += CC.total_WCSSD;
	}
}