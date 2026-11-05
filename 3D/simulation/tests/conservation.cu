// Mass conservation + periodic consistency on closed geometries with edges and corners.
// Usage: conservation [steps] [--dump DIR]; dump writes per-case CSV history and VTK checkpoints.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

static std::string slug(const std::string& name)
{
    std::string s;
    for (const char ch : name)
        s += std::isalnum((unsigned char) ch) ? (char) std::tolower((unsigned char) ch) : '_';
    return s;
}

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
static int run(const Case& c, float tau, int steps, const std::filesystem::path& dumpDir)
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

    // Cell data on voxels centred at integer lattice indices.
    auto writeVTK = [&](int step) {
        Model MD;
        std::vector<std::array<double, 4>> moments(nx * ny * nz);
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    auto& m = moments[(k * ny + j) * nx + i];
                    m = {0, 0, 0, 0};
                    if (Data->meshFluidHost(i, j, k) == 0)
                        continue;
                    for (int q = 0; q < Constants->Nvel; q++) {
                        const double f = dfHost(i, j, k, q);
                        m[0] += f;
                        for (int d = 0; d < 3; d++)
                            m[d + 1] += MD.c[q][d] * f;
                    }
                    for (int d = 1; d < 4; d++)
                        m[d] /= m[0];
                    m[0] -= 1;
                }

        std::ofstream out(dumpDir / (slug(c.name) + "_" + std::to_string(step) + ".vtk"));
        out << "# vtk DataFile Version 3.0\n" << c.name << " step " << step << "\nASCII\nDATASET STRUCTURED_POINTS\n"
            << "DIMENSIONS " << nx + 1 << ' ' << ny + 1 << ' ' << nz + 1 << "\nORIGIN -0.5 -0.5 -0.5\nSPACING 1 1 1\n"
            << "CELL_DATA " << nx * ny * nz << "\nSCALARS mesh int 1\nLOOKUP_TABLE default\n";
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    out << Data->meshFluidHost(i, j, k) << '\n';
        out << "SCALARS rho_minus_1 double 1\nLOOKUP_TABLE default\n";
        for (const auto& m : moments)
            out << m[0] << '\n';
        out << "VECTORS u double\n";
        for (const auto& m : moments)
            out << m[1] << ' ' << m[2] << ' ' << m[3] << '\n';
    };

    const bool dump = !dumpDir.empty();
    std::ofstream history;
    if (dump) {
        history.open(dumpDir / (slug(c.name) + ".csv"));
        history << "# name: " << c.name << "\n# open: " << c.openX << "\nstep,mass_drift,max_density_deviation,periodic_mismatch\n";
    }

    double mass0, mismatch;
    inspect(mass0, mismatch);
    if (dump) {
        history << 0 << ',' << 0 << ',' << maxDensityDeviation << ',' << mismatch << '\n';
        writeVTK(0);
    }

    const int checkpointEvery = std::max(1, steps / 4);
    const int sampleEvery = dump ? std::max(1, steps / 200) : checkpointEvery;
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
        const bool checkpoint = step % checkpointEvery == 0 || step == steps;
        if (!checkpoint && step % sampleEvery != 0)
            continue;
        inspect(mass, mismatch);
        finite = std::isfinite(mass);
        if (dump) {
            history << step << ',' << (mass - mass0) / mass0 << ',' << maxDensityDeviation << ',' << mismatch << '\n';
            if (checkpoint || !finite)
                writeVTK(step);
        }
        if (!finite)
            break;
    }

    const double drift = (mass - mass0) / mass0;
    // fp32 summation rounding accumulates per step; linear bound is loose on purpose.
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
    int steps = 2000;
    std::filesystem::path dumpDir;
    for (int a = 1; a < argc; a++) {
        const std::string arg = argv[a];
        if (arg == "--dump" && a + 1 < argc)
            dumpDir = argv[++a];
        else
            steps = std::stoi(arg);
    }
    if (!dumpDir.empty())
        std::filesystem::create_directories(dumpDir);

    const Case box{"box+step D3Q27", 32, 24, 20, false};
    const Case slab{"periodic-y slab+step D3Q27", 40, 10, 20, true};

    int failures = 0;
    failures += run<CollisionSRTTurbulent<Model>, OmegaNo<Model>, NoPeriodic<Model>>({box.name + " SRT", box.nx, box.ny, box.nz, false}, 0.6f, steps, dumpDir);
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, NoPeriodic<Model>>({box.name + " CUM+LES", box.nx, box.ny, box.nz, false}, 0.5005f, steps, dumpDir);
    failures += run<CollisionSRTTurbulent<Model>, OmegaNo<Model>, Periodic<Model>>({slab.name + " SRT", slab.nx, slab.ny, slab.nz, true}, 0.6f, steps, dumpDir);
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, Periodic<Model>>({slab.name + " CUM+LES", slab.nx, slab.ny, slab.nz, true}, 0.5005f, steps, dumpDir);
    const Case channel{"channel inlet/outlet+periodic-y CUM+LES", 64, 10, 24, true, true};
    failures += run<CollisionCumD3Q27Turbulent2015<Model>, OmegaLES<Model>, Periodic<Model>>(channel, 0.5005f, 4 * steps, dumpDir);
    return failures;
}
