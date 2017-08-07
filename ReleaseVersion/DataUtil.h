
#pragma once

#include <string>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::ios;
using std::cout;
using std::endl;
using std::pair;

// PointType and DcitionaryType must be the same and be float!
typedef float FloatType;
typedef FloatType PointType;
typedef FloatType DictionaryType;

typedef float QueryType;

typedef float DistanceType;
typedef int PointIdType;
typedef int CodeType;
typedef pair<DistanceType, PointIdType> DistanceToQueryType;

/**
* Compare function for DistanceToQuery.
*/
struct CompDistanceToQuery
{
	bool operator()(const DistanceToQueryType& lhs, const DistanceToQueryType& rhs)
	{
		return lhs.first < rhs.first;
	}
};

/**
* This enumeration presents different methods of kmeans algorithm.
*/
enum KmeansMethod
{
	Lloyd = 100,
	Closure
};

/**
* This enumeration presents different store types of input point coordinate.
*/
enum PointStoreType
{
	FVEC,
	BVEC,
	IVEC,
	BINARY
};

/**
 * This function read training points from point_file
 *  @param  points_file   The filename with points in .fvecs format or binary float format.
 *  @param  point_type    The type of points, should be FVECS or FLOAT.
 *  @param  points        A one-dimensional array data (of dimension*points_count).
 *  @param  points_count  The number of points.
 *  @param  dimension     The dimension of points.
 */
template<typename T>
void ReadOneDimensionalPoints(
	const string point_file, 
	PointStoreType point_sotre_type,
	vector<T>& points,
	const int points_count,
	const int dimension)
{
	ifstream point_stream;
	point_stream.open(point_file.c_str(), ios::binary);
	if (!point_stream.good())
	{
		cout << "Error in open " + point_file << endl;
		throw std::logic_error("Bad input points stream" + point_file);
	}
	int dim = 0, count = 0;
	switch (point_sotre_type)
	{
		case FVEC: 
			point_stream.read((char *)&dim, sizeof(int));
			cout << "Dimension of the vector set:" << dim << endl;
			point_stream.seekg(0, point_stream.end);
			count = point_stream.tellg() / ((dim + 1) * 4);
			cout << "Number of the vector set:" << count << endl;
			if (dim != dimension || count != points_count)
			{
				cout << "unmatched dimension!\n";
				throw std::logic_error("unmatched dimension!");
			}
			point_stream.seekg(0, point_stream.beg);
			for (int count_id = 0; count_id < count; ++count_id)
			{
				float vector_dimension = 0;
				point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
				point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(float)*dim);
			}
			break;
		case BVEC:
			point_stream.read((char *)&dim, sizeof(int));
			cout << "Dimension of the vector set:" << dim << endl;
			point_stream.seekg(0, point_stream.end);
			count = point_stream.tellg() / (dim + 4);
			cout << "Number of the vector set:" << count << endl;
			if (dim != dimension || count != points_count)
			{
				cout << "unmatched dimension!\n";
				throw std::logic_error("unmatched dimension!");
			}
			point_stream.seekg(0, point_stream.beg);
			for (int count_id = 0; count_id < count; ++count_id)
			{
				int vector_dimension = 0;
				point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
				point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(unsigned char)*dim);
			}
			break;
		case IVEC:
			point_stream.read((char *)&dim, sizeof(int));
			cout << "Dimension of the vector set:" << dim << endl;
			point_stream.seekg(0, point_stream.end);
			count = point_stream.tellg() / ((dim + 1) * 4);
			cout << "Number of the vector set:" << count << endl;
			if (dim != dimension || count != points_count)
			{
				cout << "unmatched dimension!\n";
				throw std::logic_error("unmatched dimension!");
			}
			point_stream.seekg(0, point_stream.beg);
			for (int count_id = 0; count_id < count; ++count_id)
			{
				int vector_dimension = 0;
				point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
				point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(int)*dim);
			}
			break;
		case BINARY:
			point_stream.read((char *)&count, sizeof(int));
			point_stream.read((char *)&dim, sizeof(int));
			if (dim != dimension || count != points_count)
			{
				cout << "unmatched dimension!\n";
				throw std::logic_error("unmatched dimension!");
			}
			cout << "Dimension of the vector set:" << dim << endl;
			cout << "Number of the vector set:" << count << endl;
			point_stream.read(reinterpret_cast<char*>(&(points[0])), sizeof(T)*dim*count);
			break;
	}
	point_stream.close();
}

template<typename T>
void ReadOneDimensionalPoints(
	const string point_file,
	PointStoreType point_sotre_type,
	T* points,
	const int points_count,
	const int dimension)
{
	ifstream point_stream;
	point_stream.open(point_file.c_str(), ios::binary);
	if (!point_stream.good()) {
		cout << "Error in open " + point_file << endl;
		throw std::logic_error("Bad input points stream: " + point_file);
	}
	int dim = 0, count = 0;
	switch (point_sotre_type)
	{
	case FVEC:
		point_stream.read((char *)&dim, sizeof(int));
		cout << "Dimension of the vector set:" << dim << endl;
		point_stream.seekg(0, point_stream.end);
		count = point_stream.tellg() / ((dim + 1) * 4);
		cout << "Number of the vector set:" << count << endl;
		if (dim != dimension || count != points_count)
		{
			cout << "unmatched dimension!\n";
			throw std::logic_error("unmatched dimension!");
		}
		point_stream.seekg(0, point_stream.beg);
		for (int count_id = 0; count_id < count; ++count_id)
		{
			float vector_dimension = 0;
			point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
			point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(float)*dim);
		}
		break;
	case BVEC:
		point_stream.read((char *)&dim, sizeof(int));
		cout << "Dimension of the vector set:" << dim << endl;
		point_stream.seekg(0, point_stream.end);
		count = point_stream.tellg() / (dim + 4);
		cout << "Number of the vector set:" << count << endl;
		if (dim != dimension || count != points_count)
		{
			cout << "unmatched dimension!\n";
			throw std::logic_error("unmatched dimension!");
		}
		point_stream.seekg(0, point_stream.beg);
		for (int count_id = 0; count_id < count; ++count_id)
		{
			int vector_dimension = 0;
			point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
			point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(unsigned char)*dim);
		}
		break;
	case IVEC:
		point_stream.read((char *)&dim, sizeof(int));
		cout << "Dimension of the vector set:" << dim << endl;
		point_stream.seekg(0, point_stream.end);
		count = point_stream.tellg() / ((dim + 1) * 4);
		cout << "Number of the vector set:" << count << endl;
		if (dim != dimension || count != points_count)
		{
			cout << "unmatched dimension!\n";
			throw std::logic_error("unmatched dimension!");
		}
		point_stream.seekg(0, point_stream.beg);
		for (int count_id = 0; count_id < count; ++count_id)
		{
			int vector_dimension = 0;
			point_stream.read(reinterpret_cast<char*>(&vector_dimension), sizeof(vector_dimension));
			point_stream.read(reinterpret_cast<char*>(&points[count_id*dim]), sizeof(int)*dim);
		}
		break;
	case BINARY:
		point_stream.read((char *)&count, sizeof(int));
		point_stream.read((char *)&dim, sizeof(int));
		if (dim != dimension || count != points_count)
		{
			cout << "unmatched dimension!\n";
			throw std::logic_error("unmatched dimension!");
		}
		cout << "Dimension of the vector set:" << dim << endl;
		cout << "Number of the vector set:" << count << endl;
		point_stream.read(reinterpret_cast<char*>(points), sizeof(T)*dim*count);
		break;
	}
	point_stream.close();
}

/**
* This function read training points from point_file
*  @param  points_file   The filename with points in .fvecs format or binary float format.
*  @param  points        A one-dimensional array data (of dimension*points_count).
*  @param  points_count  The number of points.
*  @param  dimension     The dimension of points.
*/
template<typename T>
void SaveOneDimensionalPoints(
	const string point_file,
	vector<T>& points,
	const int points_count,
	const int dimension)
{
	ofstream point_stream;
	point_stream.open(point_file.c_str(), ios::binary);
	if (!point_stream.good()) 
	{
		cout << "Error in write " + point_file << endl;
		throw std::logic_error("Bad output points stream" + point_file);
	}
	point_stream.write((char *)&points_count, sizeof(int));
	point_stream.write((char *)&dimension, sizeof(int));
	point_stream.write(reinterpret_cast<char*>(&(points[0])), sizeof(T)*dimension*points_count);
	point_stream.close();
}

template<typename T>
void SaveOneDimensionalPoints(
	const string point_file,
	T* points,
	const int points_count,
	const int dimension)
{
	ofstream point_stream;
	point_stream.open(point_file.c_str(), ios::binary);
	if (!point_stream.good()) 
	{
		cout << "Error in write " + point_file << endl;
		throw std::logic_error("Bad output points stream" + point_file);
	}
	point_stream.write((char *)&points_count, sizeof(int));
	point_stream.write((char *)&dimension, sizeof(int));
	point_stream.write(reinterpret_cast<char*>(points), sizeof(T)*dimension*points_count);
	point_stream.close();
}