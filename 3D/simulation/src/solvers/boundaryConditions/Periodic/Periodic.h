//
// Created by stloufra on 10/30/23.
//

#ifndef PERIODIC_H
#define PERIODIC_H

#include "../../../traits/LBMTraits.h"

// Overlap periodicity: node on one periodic plane duplicates node on partner plane (periodicIndex).
// Two passes, since partner's wall bounce-back must land before its populations are copied.
template<typename MODELDATA>
struct Periodic {
    using RealType = LBMTraits::RealType;
    using DeviceType = LBMTraits::DeviceType;
    using VectorType = LBMTraits::VectorType;
    using LBMDataPointer = TNL::Pointers::SharedPointer<LBMData, DeviceType>;
    using LBMConstantsPointer = TNL::Pointers::SharedPointer<LBMConstants, DeviceType>;

    static void periodic(LBMDataPointer &Data, LBMConstantsPointer &Constants) {

        auto periodic_view = Data->meshBoundaryPeriodic.getView();
        auto mesh_view = Data->meshFluid.getView();

        auto df_view = Data->df.getView();
        auto df_post_view = Data->df_post.getView();

        const auto Nvel = Constants->Nvel;

        MODELDATA MD;

        auto normalDotC = [=] __cuda_callable__ (const Vector &norm, int vel) mutable {
            return norm.x() * MD.c[vel][0] + norm.y() * MD.c[vel][1] + norm.z() * MD.c[vel][2];
        };

        auto bounceBack = [=]
        __cuda_callable__(
        const TNL::Containers::StaticArray<1, int> &nod) mutable
        {
            const auto bc = periodic_view[nod.x()];
            if (bc.regular)
                return;

            const Vertex vert = bc.vertex;

            for (int vel = 0; vel < Nvel; vel++) {
                if (normalDotC(bc.normal, vel) > 0)
                    continue;

                if (mesh_view(vert.x + MD.c[vel][0], vert.y + MD.c[vel][1], vert.z + MD.c[vel][2]) == 0)
                    df_view(vert.x, vert.y, vert.z, MD.c_rev[vel]) = df_post_view(vert.x, vert.y, vert.z, vel);
            }
        };

        auto copy = [=]
        __cuda_callable__(
        const TNL::Containers::StaticArray<1, int> &nod) mutable
        {
            const auto bc = periodic_view[nod.x()];
            const Vertex vert = bc.vertex;

            int partner[3] = {vert.x, vert.y, vert.z};
            if (bc.normal.x() != 0)
                partner[0] = bc.periodicIndex;
            else if (bc.normal.y() != 0)
                partner[1] = bc.periodicIndex;
            else
                partner[2] = bc.periodicIndex;

            for (int vel = 0; vel < Nvel; vel++) {
                if (normalDotC(bc.normal, vel) < 0)
                    df_view(vert.x, vert.y, vert.z, vel) = df_view(partner[0], partner[1], partner[2], vel);
            }
        };

        parallelFor<DeviceType>(0, Constants->periodic_num, bounceBack);
        parallelFor<DeviceType>(0, Constants->periodic_num, copy);
    }


};

#endif
