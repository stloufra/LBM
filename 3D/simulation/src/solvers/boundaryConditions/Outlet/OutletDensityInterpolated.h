//
// Created by stloufra on 10/30/23.
//

#ifndef OUTLETDENSITYINTERPOLATED_H
#define OUTLETDENSITYINTERPOLATED_H

#include "../../../traits/LBMTraits.h"

// Convective outflow for incoming populations, advected at cs from interior neighbour,
// then density shifted to prescribed value keeping local velocity. Normal must be axis aligned.
template<typename MODELDATA>
struct OutletDensityInterpolated {
    using RealType = LBMTraits::RealType;
    using DeviceType = LBMTraits::DeviceType;
    using VectorType = LBMTraits::VectorType;
    using LBMDataPointer = TNL::Pointers::SharedPointer<LBMData, DeviceType>;
    using LBMConstantsPointer = TNL::Pointers::SharedPointer<LBMConstants, DeviceType>;

    static void outletOmega(LBMDataPointer &Data, LBMConstantsPointer &Constants) {}

    static void outlet(LBMDataPointer &Data, LBMConstantsPointer &Constants) {

        auto outlet_view = Data->meshBoundaryOutlet.getView();
        auto mesh_view = Data->meshFluid.getView();

        auto df_view = Data->df.getView();
        auto df_post_view = Data->df_post.getView();

        const auto Nvel = Constants->Nvel;
        const RealType cs = Constants->cs;

        MODELDATA MD;

        auto f_equilibrium = [=]
        __cuda_callable__(
        RealType ux,
        RealType uy,
        RealType uz,
        RealType density,
        int vel) mutable
        {
            const RealType uc = MD.c[vel][0] * ux + MD.c[vel][1] * uy + MD.c[vel][2] * uz;
            const RealType u2 = ux * ux + uy * uy + uz * uz;
            return MD.weight[vel] * density * (1.f + 3.f * uc + 4.5f * uc * uc - 1.5f * u2);
        };

        auto bb_outlet = [=]
        __cuda_callable__(
        const TNL::Containers::StaticArray<1, int> &nod) mutable
        {
            const auto bc = outlet_view[nod.x()];
            const int i = bc.vertex.x;
            const int j = bc.vertex.y;
            const int k = bc.vertex.z;
            const int nx = bc.normal.x();
            const int ny = bc.normal.y();
            const int nz = bc.normal.z();

            for (int vel = 0; vel < Nvel; vel++) {
                const int cn = nx * MD.c[vel][0] + ny * MD.c[vel][1] + nz * MD.c[vel][2];

                if (cn < 0) {
                    df_view(i, j, k, vel) = cs * df_post_view(i - nx, j - ny, k - nz, vel) + (1 - cs) * df_post_view(i, j, k, vel);
                }
                else if (!bc.regular && mesh_view(i - MD.c[vel][0], j - MD.c[vel][1], k - MD.c[vel][2]) == 0) {
                    df_view(i, j, k, vel) = df_post_view(i, j, k, MD.c_rev[vel]);
                }
            }

            RealType rho = 0, jx = 0, jy = 0, jz = 0;
            for (int vel = 0; vel < Nvel; vel++) {
                const RealType f = df_view(i, j, k, vel);
                rho += f;
                jx += MD.c[vel][0] * f;
                jy += MD.c[vel][1] * f;
                jz += MD.c[vel][2] * f;
            }
            const RealType ux = jx / rho;
            const RealType uy = jy / rho;
            const RealType uz = jz / rho;

            for (int vel = 0; vel < Nvel; vel++) {
                df_view(i, j, k, vel) += f_equilibrium(ux, uy, uz, bc.density, vel) - f_equilibrium(ux, uy, uz, rho, vel);
            }
        };

        parallelFor<DeviceType>(0, Constants->outlet_num, bb_outlet);
    }

};

#endif
