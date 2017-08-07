#include "NoConstraintCompositeQuantization.h"

/***************************Implementation*******************************/

NoConstraintCompositeQuantization::NoConstraintCompositeQuantization(
	const int points_count,
	const int dictionaries_count,
	const int words_count,
	const int space_dimension,
	const int num_sep)
	: points_count_(points_count),
	dictionaries_count_(dictionaries_count),
	words_count_(words_count),
	space_dimension_(space_dimension),
	num_sep_(num_sep)
{
	if (dictionaries_count <= 0 || words_count <= 0 || space_dimension <= 0 || points_count <= 0 || num_sep <= 0)
	{
		cout << "NCQ: bad input parameters\n";
		throw std::logic_error("Bad input parameters");
	}
	points_ = NULL;
	own_points_memory_ = false;
	dictionary_ = new DictionaryType[dictionaries_count*words_count*space_dimension];
	memset(dictionary_, 0, sizeof(DictionaryType)*dictionaries_count*words_count*space_dimension);
	binary_codes_ = new CodeType[dictionaries_count*points_count];
	memset(binary_codes_, 0, sizeof(CodeType)*dictionaries_count*points_count);
	
	distortions_ = new float[points_count];
	memset(distortions_, 0, sizeof(float)*points_count);
	distortion_ = 0;

	binary_multi_binaryTranspose_ = new float[dictionaries_count*words_count*dictionaries_count*words_count];
	memset(binary_multi_binaryTranspose_, 0, sizeof(float)*dictionaries_count*words_count*dictionaries_count*words_count);
	binary_multi_binaryTranspose_sep_.resize(num_sep, vector<float>(dictionaries_count*words_count*dictionaries_count*words_count));
	points_multi_binaryTranspose_ = new float[dictionaries_count*words_count*space_dimension];
	memset(points_multi_binaryTranspose_, 0, sizeof(float)*dictionaries_count*words_count*space_dimension);
	points_multi_binaryTranspose_sep_.resize(num_sep, vector<float>(dictionaries_count*words_count*space_dimension));
	
	u_matrix = new float[dictionaries_count*words_count*dictionaries_count*words_count];
	s_vector = new float[dictionaries_count*words_count];
	vt_matrix = new float[dictionaries_count*words_count*dictionaries_count*words_count];
	superb = new float[dictionaries_count*words_count];
}

NoConstraintCompositeQuantization::~NoConstraintCompositeQuantization()
{
	if (superb)                        delete[] superb;
	if (vt_matrix)                     delete[] vt_matrix;
	if (s_vector)                      delete[] s_vector;
	if (u_matrix)                      delete[] u_matrix;

	if (points_multi_binaryTranspose_) delete[] points_multi_binaryTranspose_;
	if (binary_multi_binaryTranspose_) delete[] binary_multi_binaryTranspose_;
	if (distortions_)                  delete[] distortions_;
	if (binary_codes_)                 delete[] binary_codes_;
	if (dictionary_)                   delete[] dictionary_;
	if (own_points_memory_ && points_) delete[] points_;
}

void NoConstraintCompositeQuantization::SaveDictionary(const string output_file_prefix)
{
	cout << "Saving dictionary in " + output_file_prefix + "D\n";
	SaveOneDimensionalPoints<DictionaryType>(output_file_prefix + "D", dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void NoConstraintCompositeQuantization::SaveBinaryCodes(const string output_file_prefix)
{
	cout << "Saving binary codes in " + output_file_prefix + "B\n";
	SaveOneDimensionalPoints<CodeType>(output_file_prefix + "B", binary_codes_, points_count_, dictionaries_count_);
}

void NoConstraintCompositeQuantization::GetDistortions()
{
	memset(distortions_, 0, sizeof(float)*points_count_);
	int all_words_count = words_count_*dictionaries_count_;
#pragma omp parallel for
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		PointType* point = &points_[point_id*space_dimension_];
		vector<PointType> point_approximate_error(point, point + space_dimension_);
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			DictionaryType* pWord = &dictionary_[(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_];
			//PointType and DictionaryType must be the same and be float!
			cblas_saxpy(space_dimension_, -1.0, pWord, 1, &point_approximate_error[0], 1);
		}
		for (int dimension = 0; dimension < space_dimension_; ++dimension)
			distortions_[point_id] += point_approximate_error[dimension] * point_approximate_error[dimension];
	}
	distortion_ = cblas_sasum(points_count_, distortions_, 1);
}

void NoConstraintCompositeQuantization::InitPoints(
	const string points_file,
	const PointStoreType point_store_type)
{
	cout << "Reading points in " + points_file << endl;
	if (!own_points_memory_)
	{
		points_ = new PointType[space_dimension_*points_count_];
		own_points_memory_ = true;
	}
	ReadOneDimensionalPoints<PointType>(points_file, point_store_type, points_, points_count_, space_dimension_);
}

void NoConstraintCompositeQuantization::InitPoints(
	float* points,
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

void NoConstraintCompositeQuantization::InitDictionary(
	const string dictionary_file,
	const PointStoreType dictionary_store_type)
{
	cout << "Reading dictionary in " + dictionary_file << endl;
	ReadOneDimensionalPoints<DictionaryType>(dictionary_file, dictionary_store_type, dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void NoConstraintCompositeQuantization::InitDictionary(
	const float* dictionary,
	const int dictionaries_count,
	const int words_count)
{
	if (dictionaries_count != dictionaries_count_ || words_count != words_count_)
	{
		cout << "unmatched dictionary dimension\n";
		throw std::logic_error("unmatched dictionary dimension");
	}
	cout << "Reading dictionary...\n";
	memcpy(dictionary_, dictionary, sizeof(DictionaryType)*dictionaries_count_*words_count_*space_dimension_);
}

void NoConstraintCompositeQuantization::InitBinaryCodes(
	const string binary_codes_file,
	const PointStoreType binary_codes_store_type)
{
	cout << "Reading binary codes in " + binary_codes_file << endl;
	ReadOneDimensionalPoints<CodeType>(binary_codes_file, binary_codes_store_type, binary_codes_, points_count_, dictionaries_count_);
}

void NoConstraintCompositeQuantization::InitBinaryCodes(
	const CodeType* binary_codes,
	const int points_count,
	const int dictionaries_count)
{
	if (points_count != points_count_ || dictionaries_count != dictionaries_count_)
	{
		cout << "unmatched binary codes dimension\n";
		throw std::logic_error("unmatched binary codes dimension");
	}
	cout << "Reading binary codes...\n";
	memcpy(binary_codes_, binary_codes, sizeof(CodeType)*dictionaries_count_*points_count_);
}

const DictionaryType* NoConstraintCompositeQuantization::GetDictionary()
{
	return dictionary_;
}

const CodeType* NoConstraintCompositeQuantization::GetBinaryCodes()
{
	return binary_codes_;
}

void NoConstraintCompositeQuantization::InitDictionaryBinaryCodes(const string output_file_prefix)
{
	cout << "initial dictionary and binary codes using approximate results obtained from PQ...\n";
	ProductQuantization PQ(points_count_, dictionaries_count_, words_count_, space_dimension_);
	PQ.InitPoints(points_, points_count_, space_dimension_);
	PQ.Training(5, 1e-4, Lloyd, output_file_prefix + "PQ.", false);
	InitDictionary(PQ.GetDictionary(), dictionaries_count_, words_count_);
	InitBinaryCodes(PQ.GetBinaryCodes(), points_count_, dictionaries_count_);
}

void NoConstraintCompositeQuantization::Training(
	const int iters,
	const string output_file_prefix,
	const bool initial)
{
	cout << "No Constraint Composite Quantization Training...\n";
	cout << "Reminder: The points should be initialized first! \n";
	if (initial)
		InitDictionaryBinaryCodes(output_file_prefix);
	GetDistortions();
	ofstream out(output_file_prefix + "distor_iter.txt");
	if (!out.good())
	{
		cout << "Bad directory: " + output_file_prefix << endl;
		throw std::logic_error("Bad directory: " + output_file_prefix);
	}
	for (int iter = 0; iter < iters; ++iter)
	{
		cout << "Iteration " << iter << ": \n";
		out << "Iteration " << iter << ": distortion = " << distortion_ << endl;
		cout << "Updating dictionary: ";
		UpdateDictionary();
		cout << " distortion = " << distortion_ << endl;
		cout << "Updating binary codes: ";
		UpdateBinaryCodes();
		cout << " distortion = " << distortion_ << endl << endl;
	}
	out.close();
	SaveDictionary(output_file_prefix);
	SaveBinaryCodes(output_file_prefix);
}

void NoConstraintCompositeQuantization::UpdateDictionary()
{
	clock_t start = clock();
	int all_words_count = dictionaries_count_*words_count_;

	/* compute XB^T */
#pragma omp parallel for
	for (int sep = 0; sep < num_sep_; ++sep)
	{
		int start_point_id = points_count_ / num_sep_ * sep;
		int end_point_id = points_count_ / num_sep_ * (sep + 1);
		points_multi_binaryTranspose_sep_[sep].assign(space_dimension_*all_words_count, 0);
		for (int point_id = start_point_id; point_id < end_point_id; ++point_id)
		{
			CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
			PointType* point = &points_[point_id*space_dimension_];
			for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
			{
				float* points_multi_binaryTranspose_column = &(points_multi_binaryTranspose_sep_[sep][(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_]);
				// PointType must be float!
				cblas_saxpy(space_dimension_, 1.0, point, 1, points_multi_binaryTranspose_column, 1);
			}
		}
	}
	memset(points_multi_binaryTranspose_, 0, sizeof(float)*space_dimension_*all_words_count);
	for (int sep = 0; sep < num_sep_; ++sep)
	{
		cblas_saxpy(space_dimension_*all_words_count, 1.0, &(points_multi_binaryTranspose_sep_[sep][0]), 1, points_multi_binaryTranspose_, 1);
	}

	/* compute BB^T */
#pragma omp parallel for
	for (int sep = 0; sep < num_sep_; ++sep)
	{
		int start_point_id = points_count_ / num_sep_ * sep;
		int end_point_id = points_count_ / num_sep_ * (sep + 1);
		binary_multi_binaryTranspose_sep_[sep].assign(all_words_count*all_words_count, 0);
		for (int point_id = start_point_id; point_id < end_point_id; ++point_id)
		{
			CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
			PointType* point = &points_[point_id*space_dimension_];
			for (int dictionary_id_row = 0; dictionary_id_row < dictionaries_count_; ++dictionary_id_row)
			{
				int row = dictionary_id_row*words_count_ + point_codes[dictionary_id_row];
				for (int dictionary_id_col = 0; dictionary_id_col < dictionaries_count_; ++dictionary_id_col)
				{
					int col = dictionary_id_col*words_count_ + point_codes[dictionary_id_col];
					++binary_multi_binaryTranspose_sep_[sep][row*all_words_count + col];
				}
			}
		}
	}
	memset(binary_multi_binaryTranspose_, 0, sizeof(float)*all_words_count*all_words_count);
	for (int sep = 0; sep < num_sep_; ++sep)
	{
		cblas_saxpy(all_words_count*all_words_count, 1.0, &binary_multi_binaryTranspose_sep_[sep][0], 1, binary_multi_binaryTranspose_, 1);
	}

	/* singular value decomposition of (BB^T) */
	LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', all_words_count, all_words_count,
		binary_multi_binaryTranspose_, all_words_count,
		s_vector, u_matrix, all_words_count, vt_matrix, all_words_count,
		&superb[0]);

	/* compute vs^(-1) stored in vt_matrix*/
	bool zero = false;
	for (int col = 0; col < all_words_count; ++col)
	{
		if (zero || s_vector[col] < 1e-3)
		{
			zero = true;
			memset(&vt_matrix[col*all_words_count], 0, sizeof(float)*all_words_count);
			continue;
		}
		for (int row = 0; row < all_words_count; ++row)
		{
			vt_matrix[col*all_words_count + row] = vt_matrix[col*all_words_count + row] / s_vector[col];
		}
	}

	/* compute vs^(-1)u^T stored in binary_multi_binaryTranspose_ */
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, all_words_count, all_words_count, all_words_count,
		1.0, vt_matrix, all_words_count, u_matrix, all_words_count, 0, binary_multi_binaryTranspose_, all_words_count);

	/* compute XB^T(BB^T)^(-1) stored in dictionary_*/
	memset(dictionary_, 0, sizeof(DictionaryType)*space_dimension_*all_words_count);
	//DictionaryType must be float!
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, all_words_count, space_dimension_, all_words_count,
		1.0, binary_multi_binaryTranspose_, all_words_count, points_multi_binaryTranspose_, space_dimension_,
		0, dictionary_, space_dimension_);

	mkl_free_buffers();

	GetDistortions();

	clock_t finish = clock();
	cout << " cost = " << finish - start << " milliseconds \n"; 
}

void NoConstraintCompositeQuantization::UpdateBinaryCodes()
{
	clock_t start = clock();
#pragma omp parallel for
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		PointType* point = &points_[point_id*space_dimension_];

		vector<PointType> point_approximate_error(point, point + space_dimension_);
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			DictionaryType* pWord = &(dictionary_[(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_]);
			//PointType and DictioniaryType must be the same and be float!
			cblas_saxpy(space_dimension_, -1.0, pWord, 1, &point_approximate_error[0], 1);
		}

		float distortion = distortions_[point_id];
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			int old_selected_id = dictionary_id*words_count_ + point_codes[dictionary_id];
			//PointType and DictioniaryType must be the same and be float!
			cblas_saxpy(space_dimension_, 1.0, &dictionary_[old_selected_id*space_dimension_], 1, &point_approximate_error[0], 1);
			float temp_distortion;
			for (int word_id = 0; word_id < words_count_; ++word_id)
			{
				DictionaryType* pWord_temp = &dictionary_[(dictionary_id*words_count_ + word_id)*space_dimension_];
				temp_distortion = 0;
				for (int dimension = 0; dimension < space_dimension_; ++dimension)
				{
					float diff = point_approximate_error[dimension] - pWord_temp[dimension];
					temp_distortion += diff*diff;
				}
				if (temp_distortion < distortion)
				{
					distortion = temp_distortion;
					distortions_[point_id] = temp_distortion;
					point_codes[dictionary_id] = word_id;
				}
			}
			int new_selected_id = dictionary_id*words_count_ + point_codes[dictionary_id];
			//PointType and DictioniaryType must be the same and be float!
			cblas_saxpy(space_dimension_, -1.0, &dictionary_[new_selected_id*space_dimension_], 1, &point_approximate_error[0], 1);
		}
	}
	distortion_ = cblas_sasum(points_count_, distortions_, 1);
	clock_t finish = clock();
	cout << " cost = " << finish - start << " milliseconds ";
}