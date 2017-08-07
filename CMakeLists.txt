#########################################################################
# CMake build script for CompositeQuantization under Win32
#
###########################################################################
#
# start description
cmake_minimum_required (VERSION 2.8)
project (CompositeQuantization)

#########################################################################
#
# set open MP property yes
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    MESSAGE ("SET OPENMP")
endif()

MESSAGE("current dir: ${CMAKE_CURRENT_LIST_DIR}")
SET (Source_Path ${CMAKE_CURRENT_LIST_DIR}/ReleaseVersion)
SET (Current_Path ${CMAKE_CURRENT_LIST_DIR})


# add blas (this is where you need to change to your own directory)
SET(BLAS_DIR "${Current_Path}/include")
SET(BLAS_LIB "${Current_Path}/mkllib"
             "${Current_Path}/compilerlib")

########################################################################
#
# let's divide binaries in groups, for comfort navigation	
SOURCE_GROUP(Utility FILES   ${Source_Path}/DataUtil.h
                             ${Source_Path}/Kmeans.h
                             ${Source_Path}/Kmeans.cpp
                             ${Source_Path}/CQParameters.h
                             ${Source_Path}/CQParameters.cpp
                             ${Source_Path}/lbfgs.h)

	
SET(UTIL                ${Source_Path}/DataUtil.h
                        ${Source_Path}/Kmeans.h
                        ${Source_Path}/Kmeans.cpp
                        ${Source_Path}/CQParameters.h
                        ${Source_Path}/CQParameters.cpp
                        ${Source_Path}/lbfgs.h)


SOURCE_GROUP(ClosureCluster FILES   ${Source_Path}/ClosureCluster.h
                                    ${Source_Path}/ClosureCluster.cpp
                                    ${Source_Path}/Cluster.h
                                    ${Source_Path}/Cluster.cpp
                                    ${Source_Path}/ClusterCommon.h
                                    ${Source_Path}/ClusterCommon.cpp
                                    ${Source_Path}/Dataset.h
                                    ${Source_Path}/Distance.h
                                    ${Source_Path}/PartitioningTree.h
                                       ${Source_Path}/PartitioningTree.cpp)

						
						   
SET(CLOSURE             ${Source_Path}/ClosureCluster.h
                        ${Source_Path}/ClosureCluster.cpp
                        ${Source_Path}/Cluster.h
                        ${Source_Path}/Cluster.cpp
                        ${Source_Path}/ClusterCommon.h
                        ${Source_Path}/ClusterCommon.cpp
                        ${Source_Path}/Dataset.h
                        ${Source_Path}/Distance.h
                        ${Source_Path}/PartitioningTree.h
                        ${Source_Path}/PartitioningTree.cpp)

SOURCE_GROUP(Header FILES  
                      ${Source_Path}/CompositeQuantization.h
                      ${Source_Path}/NoConstraintCompositeQuantization.h
                      ${Source_Path}/ProductQuantization.h
                      ${Source_Path}/Searcher.h)

					
							
SET(HEADER            ${Source_Path}/CompositeQuantization.h
                      ${Source_Path}/NoConstraintCompositeQuantization.h
                      ${Source_Path}/ProductQuantization.h
                      ${Source_Path}/Searcher.h)


SOURCE_GROUP(Source FILES               
                    ${Source_Path}/CompositeQuantization.cpp 
                    ${Source_Path}/NoConstraintCompositeQuantization.cpp
                    ${Source_Path}/ProductQuantization.cpp
                    ${Source_Path}/Searcher.cpp
                    ${Source_Path}/Demo.cpp)


SET(SOURCE          ${Source_Path}/CompositeQuantization.cpp
                    ${Source_Path}/NoConstraintCompositeQuantization.cpp
                    ${Source_Path}/ProductQuantization.cpp
                    ${Source_Path}/Searcher.cpp 
                    ${Source_Path}/Demo.cpp)

							

# let's list all CompositeQuantization's source binaries
SET(CQTraining_ALL_CC ${UTIL} ${CLOSURE} ${HEADER} ${SOURCE})

########################################################################
#
INCLUDE_DIRECTORIES(${BLAS_DIR})
LINK_DIRECTORIES(${BLAS_LIB})
INCLUDE_DIRECTORIES(${Source_Path})
LINK_DIRECTORIES(${Current_Path}/lbfgslib)
ADD_EXECUTABLE (CompositeQuantization ${CQTraining_ALL_CC})
TARGET_LINK_LIBRARIES (CompositeQuantization  mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib lbfgs.lib)


