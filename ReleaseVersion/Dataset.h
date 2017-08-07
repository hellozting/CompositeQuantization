#pragma once
#include <stdint.h>

namespace KMC
{
    // structure to save Data Matrix
    template <typename T> 
    class Dataset 
    {
        bool ownData;  // Flag showing if the class owns its data storage.

        void shallow_copy(const Dataset& rhs)
        {
            data = rhs.data;
            rows = rhs.rows;
            cols = rhs.cols;
            ownData = false;
        }
        uint32_t rows;
        uint32_t cols;
        T* data;

    public:

	    Dataset()
	    {
	    }

	    Dataset(long rows_, long cols_, T* data_ = NULL) :
            rows(rows_), cols(cols_), data(data_), ownData(false)
	    {
            if (data_==NULL) 
		    {
		        data = new T[rows*cols];
                ownData = true;
            }
	    }

        Dataset(const Dataset& d)
        {
            shallow_copy(d);
        }

        const Dataset& operator=(const Dataset& rhs)
        {
            if (this!=&rhs) {
                shallow_copy(rhs);
            }
            return *this;
        }

	    ~Dataset()
	    {
            if (ownData) 
		    {
		      delete[] data;
            }
	    }

        /**
        * Operator that return a (pointer to a) row of the data.
        */
        T* operator[](size_t index)
        {
            return data+index*cols;
        }

        T* operator[](size_t index) const
        {
            return data+index*cols;
        }

        uint32_t R() const
        { 
            return rows; 
        }

        uint32_t C() const
        {
            return cols;
        }

    };

}