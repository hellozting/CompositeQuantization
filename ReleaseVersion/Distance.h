#pragma once  

#include <iostream>
#include "ClusterCommon.h"

namespace KMC
{
    inline FloatType ComputeDistance(const FloatType *p1, const FloatType *p2, size_t length = 100)
    {
        FloatType distance = 0;
	    for (int i = 0; i < length; i++)
	    {
            FloatType temp = p1[i] - p2[i];
            distance += temp * temp;
	    }

	    return distance;
    }

	inline FloatType ComputeDistance(const FloatType *p1, const IntegerType *p2, size_t length = 100)
    {
		FloatType distance = 0;
		for (int i = 0; i < length; i++)
		{
			FloatType temp = p1[i] - FloatType(p2[i]);
			distance += temp * temp;
	    }

	    return distance;
	}
}