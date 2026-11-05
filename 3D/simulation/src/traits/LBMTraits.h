#ifndef LBMTraits_H
#define LBMTraits_H

#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>
#include <vector>

#pragma once
using namespace TNL;
using namespace TNL::Algorithms;

struct Vertex {
    int x, y, z;
};

struct Normal {
    int x, y, z;
};

//struct Velocity {
//    double ux, uy, uz;
//};

#ifndef LBM_REAL_TYPE
#define LBM_REAL_TYPE float
#endif

using Vector = TNL::Containers::StaticVector< 3, float >;

struct boundaryConditionInlet {
    Vertex vertex;
    Vector normal;
    Vector velocity;
    bool regular;
};

struct boundaryConditionOutlet {
    Vertex vertex;
    Vector normal;
    float density;
    bool regular;
};

struct boundaryConditionSymmetry {
    Vertex vertex;
    Vector normal;
    bool regular;
};

struct boundaryConditionPeriodic {
    Vertex vertex;
    Vector normal;
    int periodicIndex;
    bool regular;
};

struct boundaryConditionWall {
    Vertex vertex;
    bool regular;
};

struct d3 {
    double x, y, z;
};

class LBMTraits {

public:


    using RealType = LBM_REAL_TYPE;

#ifdef __CUDACC__
    using DeviceType = TNL::Devices::Cuda;
#else
    using DeviceType = TNL::Devices::Host;
#endif
    using DeviceTypeHost = TNL::Devices::Host;

    using VectorType = TNL::Containers::StaticVector< 3, RealType >;
    using VectorTypeInt = TNL::Containers::StaticVector< 3, int >;
    using VectorTypeProbeCuda = TNL::Containers::Vector< RealType, DeviceType >;
    using VectorTypeProbeHost = TNL::Containers::Vector< RealType, DeviceTypeHost >;

    using TensorType = TNL::Containers::StaticVector<3, TNL::Containers::StaticVector< 3, RealType >>;

    using NDArray3DSequenceType = std::index_sequence<2, 1, 0>;
    using NDArray4DSequenceType = std::index_sequence<3, 2, 1, 0>;

    // ---------------- MESH -----------------------
    using ArrayTypeMeshHost = TNL::Containers::NDArray<         int,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceTypeHost>;

    using ArrayTypeMesh = TNL::Containers::NDArray<             int,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceType>;

    // ---------------- INLET/OUTLET -------------------
    using ArrayTypeBoundaryInletHost = TNL::Containers::NDArray<boundaryConditionInlet,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceTypeHost>;

    using ArrayTypeBoundaryOutletHost = TNL::Containers::NDArray<boundaryConditionOutlet,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceTypeHost>;


    using ArrayTypeBoundaryInlet = TNL::Containers::NDArray <   boundaryConditionInlet,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceType>;

    using ArrayTypeBoundaryOutlet = TNL::Containers::NDArray <  boundaryConditionOutlet,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceType>;

    // -------------------- WALL --------------------------
    using ArrayTypeBoundaryWall = TNL::Containers::NDArray <    boundaryConditionWall ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceType>;

    using ArrayTypeBoundaryWallHost = TNL::Containers::NDArray< boundaryConditionWall ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceTypeHost>;
    // -------------------- SYMMETRY -------------------------

    using ArrayTypeBoundarySymmetry = TNL::Containers::NDArray <    boundaryConditionSymmetry ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceType>;

    using ArrayTypeBoundarySymmetryHost = TNL::Containers::NDArray< boundaryConditionSymmetry ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceTypeHost>;

    // -------------------- PERIODIC -------------------------

    using ArrayTypeBoundaryPeriodic = TNL::Containers::NDArray <    boundaryConditionPeriodic ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceType>;

    using ArrayTypeBoundaryPeriodicHost = TNL::Containers::NDArray< boundaryConditionPeriodic ,
                                                                TNL::Containers::SizesHolder<int, 0>,
                                                                std::index_sequence<0>,
                                                                DeviceTypeHost>;


    // -------------------- VARIABLES ------------------------
    using ArrayTypeVariablesScalar = TNL::Containers::NDArray<  RealType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceType>;

    using ArrayTypeVariablesVector = TNL::Containers::NDArray<  VectorType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceType>;

    using ArrayTypeVariablesScalarHost = TNL::Containers::NDArray<  RealType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceTypeHost>;

    using ArrayTypeVariablesVectorHost = TNL::Containers::NDArray<  VectorType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0>,
                                                                NDArray3DSequenceType,
                                                                DeviceTypeHost>;

    // -------------------- DFUNCTION --------------------------
    using ArrayTypeDFunction = TNL::Containers::NDArray<        RealType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0, 0>,
                                                                NDArray4DSequenceType,
                                                                DeviceType>;

    using ArrayTypeDFunctionHost = TNL::Containers::NDArray<    RealType,
                                                                TNL::Containers::SizesHolder<int, 0, 0, 0, 0>,
                                                                NDArray4DSequenceType,
                                                                DeviceTypeHost>;


};

#endif
