#include "CompositeQuantization.h"

/***************************Implementation*******************************/

CompositeQuantization::CompositeQuantization(
	const int points_count,
	const int dictionaries_count, 
	const int words_count, 
	const int space_dimension, 
	const int num_sep)
	:points_count_(points_count), 
	dictionaries_count_(dictionaries_count),
	words_count_(words_count),
	space_dimension_(space_dimension),
	num_sep_(num_sep)
{
	if (dictionaries_count <= 0 || words_count <= 0 || space_dimension <= 0 || points_count <= 0 || num_sep <= 0)
	{
		cout << "CQ: bad input parameters\n";
		throw std::logic_error("Bad input parameters");
	}
	InitLbfgsParam();

	points_ = NULL;
	own_points_memory_ = false;
	dictionary_ = new DictionaryType[dictionaries_count*words_count*space_dimension];
	memset(dictionary_, 0, sizeof(DictionaryType)*dictionaries_count*words_count*space_dimension);
	binary_codes_ = new CodeType[dictionaries_count*points_count];
	memset(binary_codes_, 0, sizeof(CodeType)*dictionaries_count*points_count);
	
	distortions_ = new float[points_count];
	memset(distortions_, 0, sizeof(float)*points_count);
	distortion_ = 0;
	constants_ = new float[points_count];
	memset(constants_, 0, sizeof(float)*points_count);
	constant_ = 0;

	epsilon_ = 0;
	mu_ = 0;
	
	dictionary_gradient_sep_.resize(num_sep, vector<float>(dictionaries_count*words_count*space_dimension));
	dictionary_cross_products_.resize(dictionaries_count*words_count, vector<float>(dictionaries_count*words_count));
}

CompositeQuantization::~CompositeQuantization()
{
	if (constants_)    delete[] constants_;
	if (distortions_)  delete[] distortions_;
	if (binary_codes_) delete[] binary_codes_;
	if (dictionary_)   delete[] dictionary_;
	if (points_)       delete[] points_;
}

void CompositeQuantization::SaveDictionary(const string output_file_prefix)
{
	cout << "Saving dictionary in " + output_file_prefix + "D\n";
	SaveOneDimensionalPoints<DictionaryType>(output_file_prefix + "D", dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void CompositeQuantization::SaveBinaryCodes(const string output_file_prefix)
{
	cout << "Saving binary codes in " + output_file_prefix + "B\n";
	SaveOneDimensionalPoints<CodeType>(output_file_prefix + "B", binary_codes_, points_count_, dictionaries_count_);
}

void CompositeQuantization::InitPoints(
	const string points_file,
	const PointStoreType point_store_type)
{
	cout << "Reading points in " + points_file << endl;
	if (!own_points_memory_)
	{
		points_ = new float[space_dimension_*points_count_];
		own_points_memory_ = true;
	}
	ReadOneDimensionalPoints<PointType>(points_file, point_store_type, points_, points_count_, space_dimension_);
}

void CompositeQuantization::InitPoints(
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
		memcpy(points_, points, sizeof(PointType)*space_dimension_*points_count_);
	else
		points_ = points;
}

void CompositeQuantization::InitDictionary(
	const string dictionary_file,
	const PointStoreType dictionary_store_type)
{
	cout << "Reading dictionary in " + dictionary_file << endl;
	ReadOneDimensionalPoints<DictionaryType>(dictionary_file, dictionary_store_type, dictionary_, dictionaries_count_*words_count_, space_dimension_);
}

void CompositeQuantization::InitDictionary(
	const DictionaryType* dictionary,
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

void CompositeQuantization::InitBinaryCodes(
	const string binary_codes_file,
	const PointStoreType binary_codes_store_type)
{
	cout << "Reading binary codes in " + binary_codes_file << endl;
	ReadOneDimensionalPoints<CodeType>(binary_codes_file, binary_codes_store_type, binary_codes_, points_count_, dictionaries_count_);
}

void CompositeQuantization::InitBinaryCodes(
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

const DictionaryType* CompositeQuantization::GetDictionary()
{
	return dictionary_;
}

const CodeType* CompositeQuantization::GetBinaryCodes()
{
	return binary_codes_;
}

void CompositeQuantization::InitLbfgsParam()
{
	lbfgs_parameter_init(&lbfgs_param_);
	lbfgs_param_.m = 5;
}

void CompositeQuantization::InitDictionaryBinaryCodes(const string output_file_prefix)
{
	cout << "initial dictionary and binary codes using approximate results obtained from PQ...\n";
	ProductQuantization PQ(points_count_, dictionaries_count_, words_count_, space_dimension_);
	PQ.InitPoints(points_, points_count_, space_dimension_);
	PQ.Training(30, 1e-4, Closure, output_file_prefix + "PQ.", false);

	InitDictionary(PQ.GetDictionary(), dictionaries_count_, words_count_);
	InitBinaryCodes(PQ.GetBinaryCodes(), points_count_, dictionaries_count_);
}

void CompositeQuantization::GetDictionaryCrossProducts(const float* dictionary)
{
	int all_words_count = dictionaries_count_*words_count_;
#pragma omp parallel for
	for (int word_id1 = 0; word_id1 < all_words_count; ++word_id1)
	{
		for (int word_id2 = word_id1 + 1; word_id2 < all_words_count; ++word_id2)
		{
			float product = 0;
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				product += dictionary[word_id1*space_dimension_ + dimension]
							* dictionary[word_id2*space_dimension_ + dimension];
			dictionary_cross_products_[word_id1][word_id2] = product;
			dictionary_cross_products_[word_id2][word_id1] = product;
		}
	}
}

void CompositeQuantization::GetDictionaryCrossProducts(const lbfgsfloatval_t* dictionary)
{
	int all_words_count = dictionaries_count_*words_count_;
#pragma omp parallel for
	for (int word_id1 = 0; word_id1 < all_words_count; ++word_id1)
	{
		for (int word_id2 = word_id1 + 1; word_id2 < all_words_count; ++word_id2)
		{
			float product = 0;
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				product += dictionary[word_id1*space_dimension_ + dimension]
							* dictionary[word_id2*space_dimension_ + dimension];
			dictionary_cross_products_[word_id1][word_id2] = product;
			dictionary_cross_products_[word_id2][word_id1] = product;
		}
	}
}

void CompositeQuantization::GetDistortionsConstants()
{
	memset(constants_, 0, sizeof(float)*points_count_);
	memset(distortions_, 0, sizeof(float)*points_count_);

	int all_words_count = words_count_*dictionaries_count_;
#pragma omp parallel for
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		PointType* point = &points_[point_id*space_dimension_];
		vector<PointType> point_approximate_error(point, point + space_dimension_);

		float cross_product = 0;
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			int word_id1 = dictionary_id*words_count_ + point_codes[dictionary_id];
			float* pWord = &dictionary_[word_id1*space_dimension_];
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				point_approximate_error[dimension] -= pWord[dimension];
			for (int dictionary_id2 = dictionary_id + 1; dictionary_id2 < dictionaries_count_; ++dictionary_id2)
			{
				int word_id2 = dictionary_id2*words_count_ + point_codes[dictionary_id2];
				cross_product += dictionary_cross_products_[word_id1][word_id2];
			}
		}
		for (int dimension = 0; dimension < space_dimension_; ++dimension)
			distortions_[point_id] += point_approximate_error[dimension] * point_approximate_error[dimension];
		constants_[point_id] = 2 * cross_product;
	}

	distortion_ = constant_ = 0;
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		distortion_ += distortions_[point_id];
		constant_ += (constants_[point_id] - epsilon_) * (constants_[point_id] - epsilon_);
	}
}

void CompositeQuantization::UpdateEpsilon()
{
	memset(constants_, 0, sizeof(float)*points_count_);

#pragma omp parallel for
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		for (int dictionary_id1 = 0; dictionary_id1 < dictionaries_count_; ++dictionary_id1)
		{
			int word_id1 = dictionary_id1*words_count_ + point_codes[dictionary_id1];
			for (int dictionary_id2 = dictionary_id1 + 1; dictionary_id2 < dictionaries_count_; ++dictionary_id2)
			{
				int word_id2 = dictionary_id2*words_count_ + point_codes[dictionary_id2];
				constants_[point_id] += dictionary_cross_products_[word_id1][word_id2];
			}
		}
		constants_[point_id] = 2 * constants_[point_id];
	}

	float sum = 0;
	for (int point_id = 0; point_id < points_count_; ++point_id)
		sum += constants_[point_id];
	epsilon_ = sum / points_count_;
}

void CompositeQuantization::UpdateDictionary()
{
	lbfgsfloatval_t function_value;
	lbfgsfloatval_t* x = lbfgs_malloc(dictionaries_count_*words_count_*space_dimension_);
#pragma omp parallel for
	for (int i = 0; i < dictionaries_count_*words_count_*space_dimension_; ++i)
		x[i] = dictionary_[i];

	lbfgs(dictionaries_count_*words_count_*space_dimension_, x, &function_value, evaluate, progress, this, &lbfgs_param_);

#pragma omp parallel for
	for (int i = 0; i < dictionaries_count_*words_count_*space_dimension_; ++i)
		dictionary_[i] = x[i];
	lbfgs_free(x);

	GetDictionaryCrossProducts(&(dictionary_[0]));
	GetDistortionsConstants();
}

void CompositeQuantization::UpdateBinaryCodes()
{
#pragma omp parallel for
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		CodeType* point_codes = &binary_codes_[point_id*dictionaries_count_];
		PointType* point = &points_[point_id*space_dimension_];

		vector<PointType> point_approximate_error(point, point + space_dimension_);
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			DictionaryType* pWord = &(dictionary_[(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_]);
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				point_approximate_error[dimension] -= pWord[dimension];
			//PointType and DictionaryType must be the same and be float!
			//cblas_saxpy(space_dimension_, -1.0, pWord, 1, &(point_approximate_error[0]), 1);
		}

		double objective_function_value = distortions_[point_id] + mu_
			* (constants_[point_id] - epsilon_)*(constants_[point_id] - epsilon_) / 4;
		for (int dictionary_id = 0; dictionary_id < dictionaries_count_; ++dictionary_id)
		{
			//int old_selected_id = dictionary_id*words_count_ + point_codes[dictionary_id];
			DictionaryType* pWord = &(dictionary_[(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_]);
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				point_approximate_error[dimension] += pWord[dimension];
			//PointType and DictionaryType must be the same and be float!
			//cblas_saxpy(space_dimension_, 1.0, &(dictionary_[old_selected_id*space_dimension_]), 1, &(point_approximate_error[0]), 1);
			double temp_distortion, temp_constant, temp_objective_function_value;
			for (int word_id = 0; word_id < words_count_; ++word_id)
			{
				int current_selected_id = dictionary_id*words_count_ + point_codes[dictionary_id];
				int temp_selected_id = dictionary_id*words_count_ + word_id;
				DictionaryType* pWord_temp = &dictionary_[temp_selected_id*space_dimension_];
				temp_distortion = 0;
				for (int dimension = 0; dimension < space_dimension_; ++dimension)
				{
					float diff = point_approximate_error[dimension] - pWord_temp[dimension];
					temp_distortion += diff*diff;
				}
				temp_constant = constants_[point_id];
				for (int dictionary_id2 = 0; dictionary_id2 < dictionaries_count_; ++dictionary_id2)
				{
					if (dictionary_id2 == dictionary_id) continue;
					int word_id2 = dictionary_id2*words_count_ + point_codes[dictionary_id2];
					temp_constant = temp_constant + 2 * (dictionary_cross_products_[temp_selected_id][word_id2]
						- dictionary_cross_products_[current_selected_id][word_id2]);
				}
				temp_objective_function_value = temp_distortion + mu_*(temp_constant - epsilon_)*(temp_constant - epsilon_) / 4;
				if (temp_objective_function_value < objective_function_value)
				{
					objective_function_value = temp_objective_function_value;
					distortions_[point_id] = temp_distortion;
					constants_[point_id] = temp_constant;
					point_codes[dictionary_id] = word_id;
				}
			}
			//int new_selected_id = dictionary_id*words_count_ + point_codes[dictionary_id];
			pWord = &(dictionary_[(dictionary_id*words_count_ + point_codes[dictionary_id])*space_dimension_]);
			for (int dimension = 0; dimension < space_dimension_; ++dimension)
				point_approximate_error[dimension] -= pWord[dimension];
			//PointType and DictionaryType must be the same and be float!
			//cblas_saxpy(space_dimension_, -1.0, &(dictionary_[new_selected_id*space_dimension_]), 1, &point_approximate_error[0], 1);
		}
	}

	distortion_ = constant_ = 0;
	for (int point_id = 0; point_id < points_count_; ++point_id)
	{
		distortion_ += distortions_[point_id];
		constant_ += (constants_[point_id] - epsilon_) * (constants_[point_id] - epsilon_);
	}
}

void CompositeQuantization::Training(
	const int iters, 
	const double mu,
	const string output_file_prefix,
	const bool initial)
{
	mu_ = mu;
	cout << "Composite Quantization Training...\n";
	cout << "Reminder: The points, dictionary and binary codes should be initialized first! \n";

	if (initial)
		InitDictionaryBinaryCodes(output_file_prefix);
	GetDictionaryCrossProducts(&(dictionary_[0]));
	GetDistortionsConstants();

	ofstream out(output_file_prefix + "distor_iter.txt");
	for (int iter = 0; iter < iters; ++iter)
	{
		cout << "Iteration " << iter << ": distortion = " << distortion_ << ", constant = " << constant_ << endl;
		out << "Iteration " << iter << ": distortion = " << distortion_ << ", constant = " << constant_ << endl;
		cout << "Updating epsilon: \n";
		UpdateEpsilon();
		cout << "epsilon = " << epsilon_ << endl;
		cout << "Updating dictionary: \n";
		UpdateDictionary();
		cout << "Updating binary codes: \n\n";
		UpdateBinaryCodes();
	}
	out.close();

	SaveDictionary(output_file_prefix);
	SaveBinaryCodes(output_file_prefix);
}

/*****************************************************************************/
/*************************** friend function *********************************/
/*****************************************************************************/
lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step)
{
	CompositeQuantization* CQ = static_cast<CompositeQuantization*>(instance);
	int space_dimension = CQ->space_dimension_;
	CQ->GetDictionaryCrossProducts(x);

#pragma omp parallel for
	for (int sep = 0; sep < CQ->num_sep_; ++sep)
	{
		int start_point_id = CQ->points_count_ / CQ->num_sep_ * sep;
		int end_point_id = CQ->points_count_ / CQ->num_sep_ * (sep + 1);
		vector<float> apprvec(space_dimension);
		vector<float> diffvec(space_dimension);
		CQ->dictionary_gradient_sep_[sep].assign(n, 0);
		for (int point_id = start_point_id; point_id < end_point_id; ++point_id)
		{
			CodeType* point_codes = &CQ->binary_codes_[point_id*CQ->dictionaries_count_];
			PointType* point = &CQ->points_[point_id*space_dimension];
			apprvec.assign(space_dimension, 0);
			diffvec.assign(space_dimension, 0);

			float constant = 0;
			float distortion = 0;
			for (int dictionary_id = 0; dictionary_id < CQ->dictionaries_count_; ++dictionary_id)
			{
				int word_id = dictionary_id*CQ->words_count_ + point_codes[dictionary_id];
				for (int dimension = 0; dimension < space_dimension; ++dimension)
					apprvec[dimension] += x[word_id*space_dimension + dimension];
				for (int dictionary_id2 = dictionary_id + 1; dictionary_id2 < CQ->dictionaries_count_; ++dictionary_id2)
					constant += CQ->dictionary_cross_products_[word_id][dictionary_id2*CQ->words_count_ + point_codes[dictionary_id2]];
			}
			for (int dimension = 0; dimension < space_dimension; ++dimension)
			{
				float diff = apprvec[dimension] - point[dimension];
				diffvec[dimension] = diff;
				distortion += diff*diff;
			}
			CQ->distortions_[point_id] = distortion;
			CQ->constants_[point_id] = 2 * constant;

			float coeff = CQ->mu_ * (CQ->constants_[point_id] - CQ->epsilon_);
			for (int dictionary_id = 0; dictionary_id < CQ->dictionaries_count_; ++dictionary_id)
			{
				int word_id = dictionary_id*CQ->words_count_ + point_codes[dictionary_id];
				float* dictionary_gradient_column = &(CQ->dictionary_gradient_sep_[sep][word_id*space_dimension]);
				for (int dimension = 0; dimension < space_dimension; ++dimension)
					dictionary_gradient_column[dimension] += diffvec[dimension] * 2 + coeff*(apprvec[dimension] - x[word_id*space_dimension + dimension]);
			}
		}
	}

	memset(g, 0, sizeof(lbfgsfloatval_t)*n);
	for (int sep = 0; sep < CQ->num_sep_; ++sep)
	{
#pragma omp parallel for
		for (int entry_id = 0; entry_id < n; ++entry_id)
			g[entry_id] += CQ->dictionary_gradient_sep_[sep][entry_id];
	}

	CQ->distortion_ = CQ->constant_ = 0;
	for (int point_id = 0; point_id < CQ->points_count_; ++point_id)
	{
		CQ->distortion_ += CQ->distortions_[point_id];
		CQ->constant_ += (CQ->constants_[point_id] - CQ->epsilon_)*(CQ->constants_[point_id] - CQ->epsilon_);
	}

	return CQ->distortion_ + CQ->mu_ *  CQ->constant_ / 4;
}

int progress(
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
	)
{
	cout << "Lbfgs Iteration " << k << ":\n";
	cout << "  objective function value = " << fx << endl;
	cout << "  xnorm = " << xnorm << ", gnorm = " << gnorm << ", step = " << step << endl << endl;
	return 0;
}