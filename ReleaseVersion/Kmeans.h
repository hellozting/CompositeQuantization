#pragma once

#include "time.h"
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;

enum KmeansInitialType {
	KmeansInitial_RANDOM,
	KmeansInitial_KmeansPlusPlus
};

/*
 * Kmeans quantizer
 */
typedef struct _Kmeans
{
	int points_count_;                        /* The number of points. */
	int clusters_count_;                      /* The number of clusters. */
	int dimension_;                           /* The dimension of points. */
	
	float* points_;                           /* A one-dimensional array (of length dimension_*points_count_)
	                                             that the first dimension_ data is the first point. */
	float* centers_;                          /* A one-dimensional array (of length dimension_*clusters_count_)
	                                             that the first (second) dimension_ data is the first (second) center. */
	int* assignments_;                        /* A one-dimensional array (of length points_count_)
	                                             that indicate the cluster assignment of each point. */

	float* distances_;                        /* Stores the distance from each point to its assigned cluster center. */
	float distortion_;                        /* The total distortion error. */

	int max_iteration_;                       /* The maximum number of iteration. */
	double distortion_tol_;                   /* The parameter to test the distortion relative variation. */
	bool verbosity_;                          /* The flag of whether to display the current progress of the optimization process. */
}Kmeans;


/**
* This function reset the kmeans quantizer parameters.
*  @param  self             The Kmeans object.
*  @param  points_count     The number of points.
*  @param  clusters_count   The number of clusters.
*  @param  dimension        The dimension of points.
*  @param  points           A one-dimensional array (of length dimension_*points_count_)
*                           that the first dimension_ data is the first point.
*  @param  verbosity        The flag of whether to display the current progress of the optimization process.
*/
void Kmeans_Reset(
	Kmeans* self,
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	const bool verbosity = true);

/**
* This function creat a new kmeans quantizer and return it.
*  @param  points_count     The number of points.
*  @param  clusters_count   The number of clusters.
*  @param  dimension        The dimension of points.
*  @param  points           A one-dimensional array (of length dimension_*points_count_)
*                           that the first dimension_ data is the first point.
*  @param  verbosity        The flag of whether to display the current progress of the optimization process.
*/
Kmeans * Kmeans_New(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	const bool verbosity = true);

/**
* This function delete the kmeans quantizer.
*  @param  self             The Kmeans object.
*/
void Kmeans_Delete(Kmeans * self);

/**
* This function is the main function that perfrom Lloyd's algorithm.
*  @param  self             The Kmeans object.
*  @param  max_iters        The maximum iteration of the algorithm.
*  @param  distortion_tol   The parameter to test the distortion relative variation.
*/
void Kmeans_LloydQuantization(
	Kmeans* self,
	const int max_iters,
	const double distortion_tol);

/**
* This function initials the kmeans centers.
*  @param  self             The Kmeans object.
*  @param  initial_type     The way of initialize, should be RANDOM or KmeansPlusPlus.
*/
void Kmeans_Initialize(Kmeans* self, const KmeansInitialType initial_type);

/**
* This function performs random initialization.
*  @param  points_count     The number of points.
*  @param  clusters_count   The number of clusters.
*  @param  dimension        The dimension of points.
*  @param  points           A one-dimensional array (of length dimension_*points_count_)
*                           that the first dimension_ data is the first point.
*  @param  centers          A one-dimensional array (of length dimension_*clusters_count_)
*                           that the first (second) dimension_ data is the first (second) center.
*/
void Kmeans_RandomInitialize(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	float* centers);

/**
* This function performs kmeans++ initialization.
*  @param  points_count     The number of points.
*  @param  clusters_count   The number of clusters.
*  @param  dimension        The dimension of points.
*  @param  points           A one-dimensional array (of length dimension_*points_count_)
*                           that the first dimension_ data is the first point.
*  @param  centers          A one-dimensional array (of length dimension_*clusters_count_)
*                           that the first (second) dimension_ data is the first (second) center.
*/
void Kmeans_KmeansPlusPlusInitialize(
	const int points_count,
	const int clusters_count,
	const int dimension,
	const float* points,
	float* centers);