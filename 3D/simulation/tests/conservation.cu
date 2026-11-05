// Mass conservation + periodic consistency on closed geometries with edges and corners.
// Build: make -C tests (host) or make -C tests cuda.
#include <cmath>
#include <cstdio>
#include <string>

#include "../src/geometry/geometryMesherBoundary.h"
#include "../src/solvers/SolverTurbulentLES.h"

using LBMDataPointer = TNL::Pointers::SharedPointer<LBMData, LBMTraits::DeviceType>;
using LBMConstantsPointer = TNL::Pointers::SharedPointer<LBMConstants, LBMTraits::DeviceType>;
using Model = D3Q27;

struct Case {
    std::string name;
    int nx, ny, nz;
    bool periodicY;
    bool openX = false;
};

static void buildMesh(LBMDataPointer& Data, LBMConstantsPointer& Constants, const Case& c)
{
    Constants->dimX_int = c.nx;
    Constants->dimY_int = c.ny;
    Constants->dimZ_int = c.nz;
    Constants->resolution_factor = 1;
    Constants->additional_factor = 1;
    Constants->BBminx = Constants->BBminy = Constants->BBminz = 0;

    Data->meshFluidHost.setSizes(c.nx, c.ny, c.nz);
    for (int k = 0; k < c.nz; k++)
        for (int j = 0; j < c.ny; j++)
            for (int i = 0; i < c.nx; i++) {
                const bool halo = i == 0 || j == 0 || k == 0 || i == c.nx - 1 || j == c.ny - 1 || k == c.nz - 1;
                // Step obstacle on the floor: convex edge along y, two convex corners if not periodic.
                const bool step = i >= c.nx / 4 && i < c.nx / 2 && k < c.nz / 3 && (c.periodicY || (j >= c.ny / 3 && j < 2 * c.ny / 3));
                Data->meshFluidHost(i, j, k) = (halo || step) ? 0 : 1;
            }
}

template<typename Collision, typename Turbulence, typename PeriodicBC>
static int run(const Case& c, float tau, int steps)
{
    LBMConstantsPointer Constants;
    LBMDataPointer Data;
    buildMesh(Data, Constants, c);

    geometryMesherBoundary Mesher(Constants, Data);
    Mesher.meshingBoundaryWall(0);
    if (c.periodicY) {
        geometryObjectCuboid low({-1, 0.4, -1}, {-1, 0.6, c.nz + 1.}, {c.nx + 1., 0.6, -1}, -3);
        geometryObjectCuboid high({-1, c.ny - 2.6, -1}, {-1, c.ny - 2.4, c.nz + 1.}, {c.nx + 1., c.ny - 2.4, -1}, -3);
        Mesher.meshingBoundaryConditionPeriodic(low, Vector(0.f, -1.f, 0.f), c.ny - 2, 0);
        Mesher.meshingBoundaryConditionPeriodic(high, Vector(0.f, 1.f, 0.f), 1, 0);
    }
    if (c.openX) {
        geometryObjectCuboid inlet({0.4, -1, -1}, {0.6, c.ny + 1., -1}, {0.6, -1, c.nz + 1.}, -1);
        geometryObjectCuboid outlet({c.nx - 2.6, -1, -1}, {c.nx - 2.4, c.ny + 1., -1}, {c.nx - 2.4, -1, c.nz + 1.}, -2);
        Mesher.meshingBoundaryConditionInletUniform(inlet, Vector(-1.f, 0.f, 0.f), Vector(0.05f, 0.f, 0.f), 0);
        Mesher.meshingBoundaryConditionOutlet(outlet, Vector(1.f, 0.f, 0.f), 1.f, 0);
    }
    Mesher.compileBoundaryArrayWall(0);
    Mesher.compileBoundaryArrayInlets(0);
    Mesher.compileBoundaryArrayOutlets(0);
    Mesher.compileBoundaryArraySymmetry(0);
    Mesher.compileBoundaryArrayPeriodic(0);
    Mesher.arrayTransfer(0);

    Constants->Nvel = Model::numberOfDiscreteVelocities;
    Constants->tau = tau;
    Constants->omega = 1.f / tau;
    Constants->ny = (tau - 0.5f) / 3.f;
    Constants->rho_fyz = 1.f;
    Constants->Cl = Constants->Cm = Constants->Cu = Constants->Cu_inverse = 1.f;
    Constants->VelocityInit = LBMTraits::VectorType(0.05f, 0.03f, 0.02f);

    const int nx = c.nx, ny = c.ny, nz = c.nz;
    Data->rho.setSizes(nx, ny, nz);
    Data->u.setSizes(nx, ny, nz);
    Data->p.setSizes(nx, ny, nz);
    Data->omega.setSizes(nx, ny, nz);
    Data->rhoTimeAvg.setSizes(nx, ny, nz);
    Data->uTimeAvg.setSizes(nx, ny, nz);
    Data->df.setSizes(nx, ny, nz, Constants->Nvel);
    Data->df_post.setSizes(nx, ny, nz, Constants->Nvel);

    InitializationEquilibriumConstVector<Model>::initialization(Data, Constants);

    LBMTraits::ArrayTypeDFunctionHost dfHost;
    double maxDensityDeviation = 0;
    int worst[4] = {0, 0, 0, 0};
    auto inspect = [&](double& mass, double& periodicMismatch) {
        dfHost = Data->df;
        mass = 0;
        periodicMismatch = 0;
        maxDensityDeviation = 0;
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    const int m = Data->meshFluidHost(i, j, k);
                    if (m == 0)
                        continue;
                    // Overlap periodicity: plane ny-2 duplicates plane 1.
                    if (c.periodicY && j == ny - 2)
                        for (int q = 0; q < Constants->Nvel; q++)
                            periodicMismatch = std::max(periodicMismatch, (double) std::abs(dfHost(i, j, k, q) - dfHost(i, 1, k, q)));
                    else {
                        double rho = 0;
                        for (int q = 0; q < Constants->Nvel; q++)
                            rho += dfHost(i, j, k, q);
                        mass += rho;
                        if (std::abs(rho - 1) > maxDensityDeviation) {
                            maxDensityDeviation = std::abs(rho - 1);
                            worst[0] = i, worst[1] = j, worst[2] = k, worst[3] = m;
                        }
                    }
                }
    };

    double mass0, mismatch;
    inspect(mass0, mismatch);

    bool finite = true;
    double mass = mass0;
    for (int step = 1; step <= steps; step++) {
        Turbulence::omega(Data, Constants);
        Collision::collision(Data, Constants);
        StreamingAB<Model>::streaming(Data, Constants);
        BounceBackWallHalf<Model>::bounceBackWall(Data, Constants);
        PeriodicBC::periodic(Data, Constants);
        InletVelocityEquilibrium<Model>::inlet(Data, Constants);
        OutletDensityInterpolated<Model>::outlet(Data, Constants);
        MomentDensityVelocityN27<Model>::momentUpdate(Data, Constants);
        if (step % (steps / 4) == 0 || step == steps) {
            inspect(mass, mismatch);
            finite = std::isfinite(mass);
            if (!finite)
                break;
        }
    }

    const double drift = (mass - mass0) / mass0;
    // fp32 transforms round with consistent sign, so drift grows linearly in steps.
    const double tolerance = std::is_same_v<LBMTraits::RealType, double> ? 1e-10 : 1e-8 * steps;
    // Open domain: mass not conserved; velocity inlet carries pressure drop, so only require bounded density.
    const bool massOk = c.openX ? maxDensityDeviation < 0.25 : std::abs(drift) < tolerance;
    const bool ok = finite && massOk && mismatch == 0;
    std::printf("%-36s tau=%.4f steps=%d  mass drift=% .3e  max|rho-1|=%.3e  periodic mismatch=%.3e  %s\n",
                c.name.c_str(), tau, steps, drift, maxDensityDeviation, mismatch, ok ? "PASS" : "FAIL");
    if (!ok)
        std::printf("    worst density at (%d, %d, %d), mesh value %d\n", worst[0], worst[1], worst[2], worst[3]);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    const int steps = argc > 1 ? std::stoi(argv[1]) : 2000;
    const Case box{"box+step D3Q27", 32, 24, 20, false};
    const Case slab{"periodic-y slab+step D3Q27", 40, 10, 20, true};

    int failures = 0;
    failures += run<CollisionSRTTurbulent<Model>, OmegaNo<Model>, NoPeriodic<Model>>({box.name + " SRT", box.nx, box.ny, box.nz, false}, 0.6f, steps);
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, NoPeriodic<Model>>({box.name + " CUM+LES", box.nx, box.ny, box.nz, false}, 0.5005f, steps);
    failures += run<CollisionSRTTurbulent<Model>, OmegaNo<Model>, Periodic<Model>>({slab.name + " SRT", slab.nx, slab.ny, slab.nz, true}, 0.6f, steps);
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, Periodic<Model>>({slab.name + " CUM+LES", slab.nx, slab.ny, slab.nz, true}, 0.5005f, steps);
    const Case channel{"channel inlet/outlet+periodic-y CUM+LES", 64, 10, 24, true, true};
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, Periodic<Model>>(channel, 0.5005f, 4 * steps);
    return failures;
}
