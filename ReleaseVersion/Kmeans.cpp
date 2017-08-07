#include "Kmeans.h"

void Kmeans_Reset(
	Kmeans* self,
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	const bool verbosity)
{
	if (points_count*dimension > self->points_count_*self->dimension_)
	{
		if (self->points_) free(self->points_);
		self->points_ = (float*)malloc(sizeof(float)*dimension*points_count);
	}
	if (points_count > self->points_count_)
	{
		if (self->assignments_) free(self->assignments_);
		if (self->distances_) free(self->distances_);
		self->assignments_ = (int*)malloc(sizeof(int)*points_count);
		self->distances_ = (float*)malloc(sizeof(float)*points_count);
	}
	if (dimension*clusters_count > self->dimension_*self->clusters_count_)
	{
		if (self->centers_) free(self->centers_);
		self->centers_ = (float*)malloc(sizeof(float)*dimension*clusters_count);
	}
	self->points_count_ = points_count;
	self->clusters_count_ = clusters_count;
	self->dimension_ = dimension;
	self->verbosity_ = verbosity;
	if (points)
		memcpy(self->points_, points, sizeof(float)*dimension*points_count);
}

Kmeans* Kmeans_New(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	const bool verborsity)
{
	Kmeans* kmeans = (Kmeans*)malloc(sizeof(Kmeans));
	kmeans->points_ = (float*)malloc(sizeof(float)*dimension*points_count);
	kmeans->centers_ = (float*)malloc(sizeof(float)*dimension*clusters_count);
	kmeans->assignments_ = (int*)malloc(sizeof(int)*points_count);
	kmeans->distances_ = (float*)malloc(sizeof(float)*points_count);
	Kmeans_Reset(kmeans, points_count, clusters_count, dimension, points, verborsity);
	return kmeans;
}

void Kmeans_Delete(Kmeans* self)
{
	if (self->points_)
		free(self->points_);
	if (self->centers_)
		free(self->centers_);
	if (self->assignments_)
		free(self->assignments_);
	if (self->distances_)
		free(self->distances_);
	free(self);
}

void Kmeans_Initialize(Kmeans* self, const KmeansInitialType initial_type)
{
	switch (initial_type)
	{
	case KmeansInitial_RANDOM:
		Kmeans_RandomInitialize(self->points_count_, self->clusters_count_, self->dimension_, self->points_, self->centers_);
		break;
	case KmeansInitial_KmeansPlusPlus:
		Kmeans_KmeansPlusPlusInitialize(self->points_count_, self->clusters_count_, self->dimension_, self->points_, self->centers_);
	}
}

void Kmeans_RandomInitialize(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	float* centers)
{
	vector<int> perm;
	for (int id = 0; id < points_count; ++id)
		perm.push_back(id);
	std::random_shuffle(perm.begin(), perm.end());
	for (int cluster_id = 0; cluster_id < clusters_count; ++cluster_id)
	{
		memcpy(&centers[cluster_id*dimension], &points[perm[cluster_id] * dimension], sizeof(float)*dimension);
	}
}

void Kmeans_KmeansPlusPlusInitialize(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	float* centers)
{
	float* min_distances = new float[points_count];
	memset(min_distances, FLT_MAX, sizeof(float)*points_count);

	/* select the first point at random */
	int selected_id = rand() % points_count;
	int selected_ids_count = 0;
	while (true)
	{
		memcpy(&centers[selected_ids_count*dimension], &points[selected_id*dimension], sizeof(float)*dimension);
		selected_ids_count++;
		if (selected_ids_count == clusters_count) break;
		double distortion = 0;
		for (int point_id = 0; point_id < points_count; ++point_id)
		{
			float dist = 0;
			for (int dim = 0; dim < dimension; ++dim)
			{
				float diff = points[point_id*dimension + dim] - centers[(selected_ids_count - 1)*dimension + dim];
				dist += diff*diff;
			}
			if (dist < min_distances[point_id])
				min_distances[point_id] = dist;
			distortion += min_distances[point_id];
		}
		double thresh = rand() / RAND_MAX * distortion;
		double probability = 0;
		for (selected_id = 0; selected_id < points_count - 1; ++selected_id)
		{
			probability += min_distances[selected_id];
			if (probability >= thresh) break;
		}
	}
	delete[] min_distances;
}

void Kmeans_LloydQuantization(
	Kmeans* self, 
	const int max_iters, 
	const double distortion_tol)
{
	self->max_iteration_ = max_iters;
	self->distortion_tol_ = distortion_tol;
	double distortion, previous_distortion;
	int* cluster_masses = new int[self->clusters_count_];
	for (int iteration = 0; true; ++iteration)
	{
		clock_t start = clock();
		/* assign point to clusters */
#pragma omp parallel for
		for (int point_id = 0; point_id < self->points_count_; ++point_id)
		{
			float min_dist = FLT_MAX;
			int selected_id = 0;
			for (int cluster_id = 0; cluster_id < self->clusters_count_; ++cluster_id)
			{
				float dist = 0;
				for (int dim = 0; dim < self->dimension_; ++dim)
				{
					float diff = self->points_[point_id*self->dimension_ + dim] - self->centers_[cluster_id*self->dimension_ + dim];
					dist += diff*diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					selected_id = cluster_id;
				}
			}
			self->distances_[point_id] = min_dist;
			self->assignments_[point_id] = selected_id;
		}
		
		/* compute distortion*/
		distortion = 0;
		for (int point_id = 0; point_id < self->points_count_; ++point_id)
			distortion += self->distances_[point_id];
		if (self->verbosity_)
			cout << " kmeans: Lloyd iter " << iteration << " : distortion = " << distortion << endl;
		
		/* check termination conditions */
		if (iteration >= self->max_iteration_)
		{
			if (self->verbosity_)
				cout << "kmeans: Lloyd terminating because maximum number of iterations reached\n";
			break;
		}
		if (iteration == 0)
		{
			previous_distortion = distortion;
		}
		else
		{
			double eps = (previous_distortion - distortion) / previous_distortion;
			if (eps < self->distortion_tol_)
			{
				if (self->verbosity_)
					cout << "kmeans: Lloyd terminating because the distortion relative variation was less than " << self->distortion_tol_ << endl;
				break;
			}
		}

		/* begin next iteration */
		previous_distortion = distortion;

		/* update centers */
		memset(cluster_masses, 0, sizeof(int)*self->clusters_count_);
		for (int point_id = 0; point_id < self->points_count_; ++point_id)
			cluster_masses[self->assignments_[point_id]]++;

		int restarted_centers_count = 0;
		memset(self->centers_, 0, sizeof(float)*self->dimension_*self->clusters_count_);
		for (int point_id = 0; point_id < self->points_count_; ++point_id)
		{
			float* center = &self->centers_[self->assignments_[point_id] * self->dimension_];
			const float* point = &self->points_[point_id*self->dimension_];
			for (int dim = 0; dim < self->dimension_; ++dim)
			{
				center[dim] += point[dim];
			}
		}
		for (int cluster_id = 0; cluster_id < self->clusters_count_; ++cluster_id)
		{
			float* center = &self->centers_[cluster_id*self->dimension_];
			if (cluster_masses[cluster_id] > 0)
			{
				for (int dim = 0; dim < self->dimension_; ++dim)
					center[dim] /= cluster_masses[cluster_id];
			}
			else
			{
				restarted_centers_count++;
				int rand_id = rand() % self->points_count_;
				for (int dim = 0; dim < self->dimension_; ++dim)
					center[dim] = self->points_[rand_id*self->dimension_ + dim];
			}
		}
		clock_t finish = clock();
		if (self->verbosity_)
			cout << " cost = " << finish - start << " milliseconds " << endl;
	}
	self->distortion_ = distortion;
	delete[] cluster_masses;
}