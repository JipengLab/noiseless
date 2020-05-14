#include "noiseless.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <immintrin.h>
#include <chrono>

// nbody: a complicated computation procedure 

constexpr double PI = 3.141592653589793;
constexpr double SOLAR_MASS = 4 * PI * PI;
constexpr double DAYS_PER_YEAR = 365.24;

class Body
{
public:
  alignas(16) double x, vx, y, vy, z, vz;
  /*double filler;*/
  alignas(16) double mass;

  Body &offsetMomentum(const double px, const double py, const double pz) noexcept
  {
    vx = -px / SOLAR_MASS;
    vy = -py / SOLAR_MASS;
    vz = -pz / SOLAR_MASS;
    return *this;
  }
};

constexpr Body jupiter()
{
  return Body{
      /*x =*/4.84143144246472090e+00,
      /*vx =*/1.66007664274403694e-03 * DAYS_PER_YEAR,
      /*y =*/-1.16032004402742839e+00,
      /*vy =*/7.69901118419740425e-03 * DAYS_PER_YEAR,
      /*z =*/-1.03622044471123109e-01,
      /*vz =*/-6.90460016972063023e-05 * DAYS_PER_YEAR,
      /*mass =*/9.54791938424326609e-04 * SOLAR_MASS};
}

constexpr Body saturn()
{
  return Body{
      /*x = */ 8.34336671824457987e+00,
      /*vx = */ -2.76742510726862411e-03 * DAYS_PER_YEAR,
      /*y = */ 4.12479856412430479e+00,
      /*vy = */ 4.99852801234917238e-03 * DAYS_PER_YEAR,
      /*z = */ -4.03523417114321381e-01,
      /*vz = */ 2.30417297573763929e-05 * DAYS_PER_YEAR,
      /*mass =*/2.85885980666130812e-04 * SOLAR_MASS};
}

constexpr Body uranus()
{
  return Body{
      /*x =*/1.28943695621391310e+01,
      /*vx = */ 2.96460137564761618e-03 * DAYS_PER_YEAR,
      /*y =*/-1.51111514016986312e+01,
      /*vy = */ 2.37847173959480950e-03 * DAYS_PER_YEAR,
      /*z =*/-2.23307578892655734e-01,
      /*vz = */ -2.96589568540237556e-05 * DAYS_PER_YEAR,
      /*mass =*/4.36624404335156298e-05 * SOLAR_MASS};
}

constexpr Body neptune()
{
  return Body{
      /*x = */ 1.53796971148509165e+01,
      /*vx = */ 2.68067772490389322e-03 * DAYS_PER_YEAR,
      /*y = */ -2.59193146099879641e+01,
      /*vy = */ 1.62824170038242295e-03 * DAYS_PER_YEAR,
      /*z = */ 1.79258772950371181e-01,
      /*vz = */ -9.51592254519715870e-05 * DAYS_PER_YEAR,
      /*mass = */ 5.15138902046611451e-05 * SOLAR_MASS};
}

constexpr Body sun()
{
  return Body{
      /*x = */ 0,
      /*vx = */ 0,
      /*y = */ 0,
      /*vy = */ 0,
      /*z = */ 0,
      /*vz = */ 0,
      /*mass = */ SOLAR_MASS};
}

class NBodySystem
{
private:
  std::array<Body, 5> bodies;
  static constexpr unsigned int bodies_size = 5;

public:
  NBodySystem() noexcept
      : bodies{{sun(),
                jupiter(),
                saturn(),
                uranus(),
                neptune()}}
  {
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    for (unsigned i = 0; i != bodies_size; ++i)
    {
      px += bodies[i].vx * bodies[i].mass;
      py += bodies[i].vy * bodies[i].mass;
      pz += bodies[i].vz * bodies[i].mass;
    }
    bodies[0].offsetMomentum(px, py, pz);
  }

  void advance(const double dt) noexcept
  {
    constexpr unsigned N = ((bodies_size - 1) * bodies_size) >> 1;
    struct alignas(16) R
    {
      double dx, dy, dz;
    };
    static R r[1000];
    alignas(16) static double mag[1000];

    for (unsigned int i = 0, k = 0; i != bodies_size - 1; ++i)
    {
      for (unsigned int j = i + 1; j != bodies_size; ++j, ++k)
      {
        r[k].dx = bodies[i].x - bodies[j].x;
        r[k].dy = bodies[i].y - bodies[j].y;
        r[k].dz = bodies[i].z - bodies[j].z;
      }
    }

    for (unsigned int i = 0; i != N; i += 2)
    {
      __m128d const dx = _mm_loadh_pd(_mm_loadl_pd(dx, &r[i].dx), &r[i + 1].dx);
      __m128d const dy = _mm_loadh_pd(_mm_loadl_pd(dy, &r[i].dy), &r[i + 1].dy);
      __m128d const dz = _mm_loadh_pd(_mm_loadl_pd(dz, &r[i].dz), &r[i + 1].dz);

      //__m128d dSquared = dx*dx + dy*dy + dz*dz;
      __m128d const dSquared = _mm_add_pd(
          _mm_add_pd(_mm_mul_pd(dx, dx), _mm_mul_pd(dy, dy)),
          _mm_mul_pd(dz, dz));

      __m128d distance =
          _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(dSquared)));

      for (int j = 0; j != 2; ++j)
      {
        distance = _mm_sub_pd(
            _mm_mul_pd(distance, _mm_set1_pd(1.5)),
            _mm_mul_pd(
                _mm_mul_pd(
                    _mm_mul_pd(_mm_set1_pd(0.5), dSquared),
                    distance),
                _mm_mul_pd(distance, distance)));
      }

      __m128d const dmag = _mm_mul_pd(
          _mm_div_pd(_mm_set1_pd(dt), dSquared), distance);
      _mm_store_pd(&mag[i], dmag);
    }

    for (unsigned int i = 0, k = 0; i != bodies_size - 1; ++i)
    {
      Body &iBody = bodies[i];
      for (unsigned int j = i + 1; j != bodies_size; ++j, ++k)
      {
        Body &jBody = bodies[j];
        double const jmm = jBody.mass * mag[k];
        iBody.vx -= r[k].dx * jmm;
        iBody.vy -= r[k].dy * jmm;
        iBody.vz -= r[k].dz * jmm;

        double const imm = iBody.mass * mag[k];
        jBody.vx += r[k].dx * imm;
        jBody.vy += r[k].dy * imm;
        jBody.vz += r[k].dz * imm;
      }
    }

    for (unsigned int i = 0; i != bodies_size; ++i)
    {
      Body &iBody = bodies[i];
      iBody.x += dt * iBody.vx;
      iBody.y += dt * iBody.vy;
      iBody.z += dt * iBody.vz;
    }
  }

  double energy() noexcept
  {
    double e = 0.0;

    for (unsigned int i = 0; i != bodies_size; ++i)
    {
      Body const iBody = bodies[i];
      e += 0.5 * iBody.mass *
           (iBody.vx * iBody.vx + iBody.vy * iBody.vy + iBody.vz * iBody.vz);

      for (unsigned int j = i + 1; j != bodies_size; ++j)
      {
        Body const &jBody = bodies[j];

        double const dx = iBody.x - jBody.x;
        double const dy = iBody.y - jBody.y;
        double const dz = iBody.z - jBody.z;

        double const distance = pow(dx * dx + dy * dy + dz * dz, 0.5);
        e -= (iBody.mass * jBody.mass) / distance;
      }
    }
    return e;
  }
};

double nbody()
{
  NBodySystem bodies;
  std::cout << std::setprecision(9) << bodies.energy() << '\n';
  for (int i = 0; i< 500000; i++) bodies.advance(0.01);
  return bodies.energy();
}

int main()
{
  noiseless::Noiseless::Instance().SetCpuAffinityToNotIsolatedCpu();
  auto res = noiseless::Noiseless::Instance().RunRepeatedlyWithoutNoiseless(10, nbody);
  std::cout << res.ToString() << std::endl;
  return 0;
}