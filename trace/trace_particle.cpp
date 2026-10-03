#include <fusion_io_defs.h>
#include <fusion_io_field.h>
#include <m3dc1_source.h>
#include <options.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double elementary_charge = 1.602176634e-19;
constexpr double proton_mass = 1.67262192369e-27;
constexpr double kev_to_joule = 1.0e3 * elementary_charge;
constexpr double pi = 3.14159265358979323846;

using Vec3 = std::array<double, 3>;
using State = std::array<double, 4>; // R, phi, Z, v_parallel

struct MotionPeriods {
  double toroidal = std::numeric_limits<double>::quiet_NaN();
  double poloidal = std::numeric_limits<double>::quiet_NaN();
};

struct MotionPeriodTracker {
  MotionPeriods periods;
  double axis_r;
  double axis_z;
  double start_time = 0.0;
  double start_phi = 0.0;
  double initial_theta = 0.0;
  double previous_theta_offset = 0.0;
  double toroidal_displacement = 0.0;
  int poloidal_departure_direction = 0;
  int initial_parallel_sign = 0;
  bool parallel_reversed = false;
  bool poloidal_initialized = false;

  MotionPeriodTracker(double r_axis, double z_axis)
    : axis_r(r_axis), axis_z(z_axis)
  {
  }
};

struct ScanRange {
  double minimum = std::numeric_limits<double>::quiet_NaN();
  double maximum = std::numeric_limits<double>::quiet_NaN();
  int count = 0;

  bool enabled() const { return count > 0; }
};

enum ScanStatus {
  scan_complete = 0,
  scan_incomplete = 1,
  scan_invalid_energy = 2,
  scan_lost_or_singular = 3,
  scan_invalid_pphi = 4,
  scan_output_error = 5
};

enum OrbitType {
  orbit_unknown = 0,
  orbit_passing = 1,
  orbit_trapped = 2
};

struct ScanResult {
  int index = 0;
  int steps = 0;
  int status = scan_complete;
  int sigma = 1;
  int orbit_type = orbit_unknown;
  double energy = 0.0;
  double mu = 0.0;
  double lambda = std::numeric_limits<double>::quiet_NaN();
  double pphi = std::numeric_limits<double>::quiet_NaN();
  double initial_r = std::numeric_limits<double>::quiet_NaN();
  double toroidal_period = std::numeric_limits<double>::quiet_NaN();
  double poloidal_period = std::numeric_limits<double>::quiet_NaN();
  double orbit_period = std::numeric_limits<double>::quiet_NaN();
  double jacobian_relative = std::numeric_limits<double>::quiet_NaN();
};

struct Parameters {
  std::string filename = "C1.h5";
  std::string output = "particle_orbit.out";
  int timeslice = -1;
  int steps = 10000;
  int output_every = 1;
  int sigma = 1;
  int nplanes = 1;
  int transits = 0;
  double scale = 1.0;
  double phase = 0.0;
  double poincare_plane = 0.0;
  double r = std::numeric_limits<double>::quiet_NaN();
  double phi = 0.0;
  double z = std::numeric_limits<double>::quiet_NaN();
  double energy_kev = std::numeric_limits<double>::quiet_NaN();
  double mu_kev_per_t = std::numeric_limits<double>::quiet_NaN();
  double lambda = std::numeric_limits<double>::quiet_NaN();
  double pphi = std::numeric_limits<double>::quiet_NaN();
  ScanRange energy_range;
  ScanRange mu_range;
  ScanRange lambda_range;
  ScanRange pphi_range;
  double dt = 1.0e-9;
  double mass_ratio = std::numeric_limits<double>::quiet_NaN();
  double charge_state = std::numeric_limits<double>::quiet_NaN();
  bool equilibrium = true;
  bool perturbed = true;
  bool pout = false;
  bool qout = true;
  bool output_set = false;
};

struct Fields {
  m3dc1_source source;
  fio_field* magnetic = nullptr;
  fio_field* equilibrium_magnetic = nullptr;
  fio_field* poloidal_flux = nullptr;
  fio_hint hint = nullptr;
  double period = 2.0 * pi;

  ~Fields()
  {
    if(hint) source.deallocate_search_hint(&hint);
    delete magnetic;
    delete equilibrium_magnetic;
    delete poloidal_flux;
    source.close();
  }
};

void print_help(const char* program)
{
  std::cout
    << "Usage: " << program << " -m3dc1 FILE\n"
    << "       --energy {VALUE | MIN MAX N}\n"
    << "       {--mu | --lambda} {VALUE | MIN MAX N} [options]\n"
    << "\n"
    << "Trace one guiding-center orbit using the equations implemented in particle.f90.\n"
    << "The initial position defaults to the magnetic axis when R or Z is omitted.\n"
    << "\n"
    << "Particle or scan coordinates:\n"
    << "  --energy VALUE | MIN MAX N  Energy [keV], scalar or N-point scan\n"
    << "  --mu VALUE | MIN MAX N      Magnetic moment [keV/T], scalar or N-point scan\n"
    << "  --lambda VALUE | MIN MAX N  Lambda=mu*B0/E, scalar or N-point scan\n"
    << "  --pphi VALUE | MIN MAX N    P_phi/q [Wb], scalar or N-point scan\n"
    << "  P_phi scans solve their starting R on the low-field-side midplane.\n"
    << "\n"
    << "Field and particle options:\n"
    << "  -m3dc1 FILE          M3D-C1 HDF5 file (default: C1.h5)\n"
    << "  --timeslice N        Field timeslice (default: -1, equilibrium)\n"
    << "  --equilibrium {0,1}  Include the equilibrium magnetic field (default: 1)\n"
    << "  --perturbed {0,1}    Include the perturbed magnetic field (default: 1)\n"
    << "  --factor VALUE       Perturbed-field multiplication factor (default: 1)\n"
    << "  --scale VALUE        Alias for --factor\n"
    << "  --phase VALUE        Toroidal phase shift [degrees] (default: 0)\n"
    << "  --mass VALUE         Mass in proton-mass units (default: file ion_mass)\n"
    << "  --charge VALUE       Charge state in units of e (default: file z_ion)\n"
    << "  --sigma {-1,0,1}     Initial sign of v_parallel; 0 scans both (default: 1)\n"
    << "\n"
    << "Orbit options:\n"
    << "  --R VALUE            Initial major radius [m]\n"
    << "  --phi VALUE          Initial toroidal angle [degrees] (default: 0)\n"
    << "  --Z VALUE            Initial vertical coordinate [m]\n"
    << "  --dt VALUE           RK4 time step [s] (default: 1e-9)\n"
    << "  --steps N            Number of time steps (default: 10000)\n"
    << "  --output-every N     Write every Nth step (default: 1)\n"
    << "  --output FILE        Output table (default: particle_orbit.out, or\n"
    << "                       particle_period_scan.out in scan mode)\n"
    << "                       MPI ranks divide scan points cyclically\n"
    << "  -pout {0,1}         Write Poincare crossings to out<index> (default: 0)\n"
    << "  -qout {0,1}         Calculate and write motion periods (default: 1)\n"
    << "  -t VALUE            Stop after this many toroidal transits (0: disabled)\n"
    << "  -a VALUE            Poincare plane angle [degrees] (default: 0)\n"
    << "  -n VALUE            Number of equally spaced output planes (default: 1)\n"
    << "  -h, --help           Show this help\n";
}

double parse_double(const char* text, const std::string& option)
{
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if(!end || *end != '\0')
    throw std::runtime_error("Invalid value for " + option + ": " + text);
  return value;
}

int parse_int(const char* text, const std::string& option)
{
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if(!end || *end != '\0')
    throw std::runtime_error("Invalid value for " + option + ": " + text);
  return static_cast<int>(value);
}

const char* require_value(int& i, int argc, char* argv[], const std::string& option)
{
  if(++i >= argc) throw std::runtime_error("Missing value for " + option);
  return argv[i];
}

bool parse_numeric_token(const char* text, double& value)
{
  char* end = nullptr;
  value = std::strtod(text, &end);
  return end && *end == '\0';
}

void parse_value_or_range(int& i, int argc, char* argv[],
                          const std::string& option, double& value,
                          ScanRange& range)
{
  const double first = parse_double(require_value(i, argc, argv, option), option);
  double maximum = 0.0;
  if(i + 1 >= argc || !parse_numeric_token(argv[i + 1], maximum)) {
    value = first;
    return;
  }
  if(i + 2 >= argc)
    throw std::runtime_error(option + " range requires MIN MAX N");

  const int count = parse_int(argv[i + 2], option);
  if(count < 1)
    throw std::runtime_error(option + " range N must be positive");
  range.minimum = first;
  range.maximum = maximum;
  range.count = count;
  value = std::numeric_limits<double>::quiet_NaN();
  i += 2;
}

bool parse_command_line(int argc, char* argv[], Parameters& p, bool show_help)
{
  for(int i = 1; i < argc; ++i) {
    const std::string option(argv[i]);
    if(option == "-h" || option == "--help") {
      if(show_help) print_help(argv[0]);
      return false;
    } else if(option == "-m3dc1") {
      p.filename = require_value(i, argc, argv, option);
    } else if(option == "--timeslice") {
      p.timeslice = parse_int(require_value(i, argc, argv, option), option);
    } else if(option == "--factor" || option == "--scale") {
      p.scale = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--equilibrium") {
      const int enabled = parse_int(require_value(i, argc, argv, option), option);
      if(enabled != 0 && enabled != 1)
        throw std::runtime_error("--equilibrium must be 0 or 1");
      p.equilibrium = enabled == 1;
    } else if(option == "--perturbed") {
      const int enabled = parse_int(require_value(i, argc, argv, option), option);
      if(enabled != 0 && enabled != 1)
        throw std::runtime_error("--perturbed must be 0 or 1");
      p.perturbed = enabled == 1;
    } else if(option == "--phase") {
      p.phase = parse_double(require_value(i, argc, argv, option), option) * pi / 180.0;
    } else if(option == "--R" || option == "--r") {
      p.r = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--phi") {
      p.phi = parse_double(require_value(i, argc, argv, option), option) * pi / 180.0;
    } else if(option == "--Z" || option == "--z") {
      p.z = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--energy") {
      parse_value_or_range(i, argc, argv, option, p.energy_kev, p.energy_range);
    } else if(option == "--mu") {
      parse_value_or_range(i, argc, argv, option, p.mu_kev_per_t, p.mu_range);
    } else if(option == "--lambda") {
      parse_value_or_range(i, argc, argv, option, p.lambda, p.lambda_range);
    } else if(option == "--pphi") {
      parse_value_or_range(i, argc, argv, option, p.pphi, p.pphi_range);
    } else if(option == "--dt") {
      p.dt = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--steps") {
      p.steps = parse_int(require_value(i, argc, argv, option), option);
    } else if(option == "--output-every") {
      p.output_every = parse_int(require_value(i, argc, argv, option), option);
    } else if(option == "--output") {
      p.output = require_value(i, argc, argv, option);
      p.output_set = true;
    } else if(option == "-pout" || option == "--pout") {
      p.pout = parse_int(require_value(i, argc, argv, option), option) != 0;
    } else if(option == "-qout" || option == "--qout") {
      p.qout = parse_int(require_value(i, argc, argv, option), option) != 0;
    } else if(option == "-t" || option == "--transits") {
      p.transits = parse_int(require_value(i, argc, argv, option), option);
    } else if(option == "-a" || option == "--poincare-plane") {
      p.poincare_plane = parse_double(require_value(i, argc, argv, option), option)
                          * pi / 180.0;
    } else if(option == "-n" || option == "--nplanes") {
      p.nplanes = parse_int(require_value(i, argc, argv, option), option);
    } else if(option == "--mass" || option == "--mass-ratio") {
      p.mass_ratio = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--charge") {
      p.charge_state = parse_double(require_value(i, argc, argv, option), option);
    } else if(option == "--sigma") {
      p.sigma = parse_int(require_value(i, argc, argv, option), option);
    } else {
      throw std::runtime_error("Unknown option: " + option);
    }
  }

  if(p.energy_range.enabled()) {
    if(!std::isfinite(p.energy_range.minimum)
       || !std::isfinite(p.energy_range.maximum)
       || p.energy_range.minimum <= 0.0
       || p.energy_range.maximum < p.energy_range.minimum)
      throw std::runtime_error("--energy range requires 0 < MIN <= MAX");
  } else if(!std::isfinite(p.energy_kev) || p.energy_kev <= 0.0) {
    throw std::runtime_error("Specify a positive --energy");
  }
  const bool mu_given = p.mu_range.enabled() || std::isfinite(p.mu_kev_per_t);
  const bool lambda_given = p.lambda_range.enabled() || std::isfinite(p.lambda);
  if(mu_given && lambda_given)
    throw std::runtime_error("Use either --mu or --lambda, not both");
  if(!mu_given && !lambda_given)
    throw std::runtime_error("Specify --mu or --lambda");
  if(p.mu_range.enabled()) {
    if(!std::isfinite(p.mu_range.minimum)
       || !std::isfinite(p.mu_range.maximum)
       || p.mu_range.minimum < 0.0
       || p.mu_range.maximum < p.mu_range.minimum)
      throw std::runtime_error("--mu range requires 0 <= MIN <= MAX");
  } else if(mu_given && (!std::isfinite(p.mu_kev_per_t)
                         || p.mu_kev_per_t < 0.0)) {
    throw std::runtime_error("Specify a nonnegative --mu");
  }
  if(p.lambda_range.enabled()) {
    if(!std::isfinite(p.lambda_range.minimum)
       || !std::isfinite(p.lambda_range.maximum)
       || p.lambda_range.minimum < 0.0
       || p.lambda_range.maximum < p.lambda_range.minimum)
      throw std::runtime_error("--lambda range requires 0 <= MIN <= MAX");
  } else if(lambda_given && (!std::isfinite(p.lambda) || p.lambda < 0.0)) {
    throw std::runtime_error("Specify a nonnegative --lambda");
  }
  if(p.pphi_range.enabled()
     && (!std::isfinite(p.pphi_range.minimum)
         || !std::isfinite(p.pphi_range.maximum)
         || p.pphi_range.maximum < p.pphi_range.minimum))
    throw std::runtime_error("--pphi range requires finite MIN <= MAX");
  if((p.pphi_range.enabled() || std::isfinite(p.pphi))
     && (std::isfinite(p.r) || std::isfinite(p.z)))
    throw std::runtime_error("Do not specify R or Z for a P_phi scan");
  if(p.sigma != -1 && p.sigma != 0 && p.sigma != 1)
    throw std::runtime_error("--sigma must be -1, 0, or 1");
  if(!p.equilibrium && !p.perturbed)
    throw std::runtime_error("At least one of --equilibrium or --perturbed must be enabled");
  if(p.dt <= 0.0 || p.steps < 1 || p.output_every < 1)
    throw std::runtime_error("--dt, --steps, and --output-every must be positive");
  if(p.nplanes < 1)
    throw std::runtime_error("-n/--nplanes must be positive");
  if(p.transits < 0)
    throw std::runtime_error("-t/--transits must be nonnegative");
  if(!p.pout && !p.qout)
    throw std::runtime_error("At least one of -pout or -qout must be enabled");
  if(std::isfinite(p.charge_state) && p.charge_state == 0.0)
    throw std::runtime_error("--charge must be nonzero");
  if((p.energy_range.enabled() || p.mu_range.enabled() || p.lambda_range.enabled()
      || p.pphi_range.enabled() || std::isfinite(p.pphi) || p.sigma == 0)
     && !p.output_set)
    p.output = "particle_period_scan.out";
  return true;
}

double wrap_phi(double phi, double period)
{
  phi = std::fmod(phi, period);
  return phi < 0.0 ? phi + period : phi;
}

double signed_angle_difference(double angle, double reference)
{
  return std::remainder(angle - reference, 2.0 * pi);
}

int parallel_sign(double value)
{
  return value > 0.0 ? 1 : (value < 0.0 ? -1 : 0);
}

void initialize_poloidal_period(const State& state, double time,
                                MotionPeriodTracker& tracker)
{
  if(tracker.initial_parallel_sign == 0)
    tracker.initial_parallel_sign = parallel_sign(state[3]);
  const double dr = state[0] - tracker.axis_r;
  const double dz = state[2] - tracker.axis_z;
  if(std::hypot(dr, dz) <= 1.0e-12) return;

  tracker.initial_theta = std::atan2(dz, dr);
  tracker.previous_theta_offset = 0.0;
  tracker.start_time = time;
  tracker.start_phi = state[1];
  tracker.poloidal_initialized = true;
}

// Determine first full toroidal transit and first same-direction return in
// poloidal angle. The latter is also the bounce period for a trapped orbit.
void calculate_motion_period(const State& previous, const State& current,
                             double previous_time, double current_time,
                             MotionPeriodTracker& tracker)
{
  const int current_parallel_sign = parallel_sign(current[3]);
  if(tracker.initial_parallel_sign == 0 && current_parallel_sign != 0)
    tracker.initial_parallel_sign = current_parallel_sign;
  else if(current_parallel_sign != 0
          && current_parallel_sign != tracker.initial_parallel_sign)
    tracker.parallel_reversed = true;

  if(!std::isfinite(tracker.periods.toroidal)) {
    const double old_displacement = tracker.toroidal_displacement;
    tracker.toroidal_displacement += current[1] - previous[1];
    if(std::abs(tracker.toroidal_displacement) >= 2.0 * pi) {
      const double denominator = std::abs(tracker.toroidal_displacement)
                                 - std::abs(old_displacement);
      const double fraction = denominator > 0.0
        ? (2.0 * pi - std::abs(old_displacement)) / denominator : 1.0;
      tracker.periods.toroidal = previous_time
        + fraction * (current_time - previous_time);
    }
  }

  if(std::isfinite(tracker.periods.poloidal)) return;
  if(!tracker.poloidal_initialized) {
    initialize_poloidal_period(current, current_time, tracker);
    return;
  }

  const double dr = current[0] - tracker.axis_r;
  const double dz = current[2] - tracker.axis_z;
  if(std::hypot(dr, dz) <= 1.0e-12) return;

  const double theta = std::atan2(dz, dr);
  const double theta_offset = signed_angle_difference(theta, tracker.initial_theta);
  if(tracker.poloidal_departure_direction == 0
     && std::abs(theta_offset) > 1.0e-10) {
    tracker.poloidal_departure_direction = theta_offset > 0.0 ? 1 : -1;
  }

  const double offset_change = theta_offset - tracker.previous_theta_offset;
  const bool crosses_initial_ray = tracker.previous_theta_offset * theta_offset <= 0.0
    && std::abs(offset_change) < pi
    && std::abs(tracker.previous_theta_offset) > 1.0e-10;
  const bool same_direction = tracker.poloidal_departure_direction * offset_change > 0.0;
  if(crosses_initial_ray && same_direction) {
    const double fraction = -tracker.previous_theta_offset / offset_change;
    const double crossing_time = previous_time
      + fraction * (current_time - previous_time);
    const double crossing_phi = previous[1]
      + fraction * (current[1] - previous[1]);
    tracker.periods.poloidal = crossing_time - tracker.start_time;
    const double toroidal_advance = std::abs(crossing_phi - tracker.start_phi);
    if(toroidal_advance > 1.0e-12)
      tracker.periods.toroidal = tracker.periods.poloidal * 2.0 * pi
                                 / toroidal_advance;
  }
  tracker.previous_theta_offset = theta_offset;
}

int detected_orbit_type(const MotionPeriodTracker& tracker)
{
  if(tracker.parallel_reversed) return orbit_trapped;
  if(std::isfinite(tracker.periods.poloidal)) return orbit_passing;
  return orbit_unknown;
}

const char* orbit_type_name(int type)
{
  switch(type) {
  case orbit_passing: return "passing";
  case orbit_trapped: return "trapped";
  default:            return "unknown";
  }
}

double relative_jacobian(double orbit_period, double energy_kev,
                         double reference_b)
{
  if(!std::isfinite(orbit_period) || !std::isfinite(reference_b)
     || reference_b <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return orbit_period * energy_kev / reference_b;
}

struct PoincarePoint {
  double fraction;
  double phi;
};

void write_poincare_crossings(std::ostream& out, const State& previous,
                              const State& current, double previous_time,
                              double current_time, double plane, int nplanes,
                              double axis_r, double axis_z)
{
  const double delta_phi = current[1] - previous[1];
  if(delta_phi == 0.0) return;

  const double minimum_phi = std::min(previous[1], current[1]);
  const double maximum_phi = std::max(previous[1], current[1]);
  std::vector<PoincarePoint> crossings;
  for(int output_plane = 0; output_plane < nplanes; ++output_plane) {
    const double base = plane + 2.0 * pi * output_plane / nplanes;
    const long first = static_cast<long>(std::ceil((minimum_phi - base)
                                                   / (2.0 * pi)));
    const long last = static_cast<long>(std::floor((maximum_phi - base)
                                                  / (2.0 * pi)));
    for(long transit = first; transit <= last; ++transit) {
      const double crossing_phi = base + 2.0 * pi * transit;
      const double fraction = (crossing_phi - previous[1]) / delta_phi;
      if(fraction > 1.0e-12 && fraction <= 1.0)
        crossings.push_back({fraction, crossing_phi});
    }
  }
  std::sort(crossings.begin(), crossings.end(),
            [](const PoincarePoint& a, const PoincarePoint& b) {
              return a.fraction < b.fraction;
            });

  for(const PoincarePoint& crossing : crossings) {
    const double r = previous[0]
                     + crossing.fraction * (current[0] - previous[0]);
    const double z = previous[2]
                     + crossing.fraction * (current[2] - previous[2]);
    const double time = previous_time
                        + crossing.fraction * (current_time - previous_time);
    const double theta = std::atan2(z - axis_z, r - axis_r) * 180.0 / pi;
    out << wrap_phi(crossing.phi, 2.0 * pi) * 180.0 / pi << ' '
        << r << ' ' << z << ' ' << theta << ' ' << time << '\n';
  }
}

double norm(const Vec3& a)
{
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

double dot(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

bool load_fields(const Parameters& p, Fields& fields)
{
  if(fields.source.open(p.filename.c_str()) != FIO_SUCCESS) {
    std::cerr << "Error opening " << p.filename << '\n';
    return false;
  }

  fio_option_list options;
  fields.source.get_field_options(&options);
  options.set_option(FIO_TIMESLICE, p.timeslice);
  options.set_option(FIO_LINEAR_SCALE, p.scale);
  options.set_option(FIO_PHASE, p.phase);
  if(p.equilibrium && p.perturbed)
    options.set_option(FIO_PART, FIO_TOTAL);
  else if(p.equilibrium)
    options.set_option(FIO_PART, FIO_EQUILIBRIUM_ONLY);
  else
    options.set_option(FIO_PART, FIO_PERTURBED_ONLY);

  if(fields.source.get_field(FIO_MAGNETIC_FIELD, &fields.magnetic, &options)
      != FIO_SUCCESS) {
    std::cerr << "Error loading the magnetic field\n";
    return false;
  }
  if(p.pphi_range.enabled() || std::isfinite(p.pphi)) {
    options.set_option(FIO_PART, FIO_EQUILIBRIUM_ONLY);
    if(fields.source.get_field(FIO_MAGNETIC_FIELD,
                               &fields.equilibrium_magnetic, &options)
         != FIO_SUCCESS
       || fields.source.get_field(FIO_POLOIDAL_FLUX,
                                  &fields.poloidal_flux, &options)
            != FIO_SUCCESS) {
      std::cerr << "Error loading equilibrium fields for the P_phi scan\n";
      return false;
    }
  }
  if(fields.source.allocate_search_hint(&fields.hint) != FIO_SUCCESS)
    fields.hint = nullptr;
  fields.source.get_real_parameter(FIO_PERIOD, &fields.period);
  return true;
}

bool magnetic_data(Fields& fields, const State& state, Vec3& b,
                   std::array<double, 9>& db)
{
  const double x[3] = {state[0], wrap_phi(state[1], fields.period), state[2]};
  return fields.magnetic->eval(x, b.data(), fields.hint) == FIO_SUCCESS
      && fields.magnetic->eval_deriv(x, db.data(), fields.hint) == FIO_SUCCESS;
}

bool equilibrium_values(Fields& fields, double r, double phi, double z,
                        Vec3& b, double* psi)
{
  const double x[3] = {r, wrap_phi(phi, fields.period), z};
  double flux = 0.0;
  if(fields.equilibrium_magnetic->eval(x, b.data(), fields.hint) != FIO_SUCCESS
     || fields.poloidal_flux->eval(x, &flux, fields.hint) != FIO_SUCCESS)
    return false;
  *psi = flux / (2.0 * pi);
  return true;
}

bool find_lfs_edge(Fields& fields, double axis_r, double axis_z, double phi,
                   double psi_lcfs, double* edge_r)
{
  constexpr double inside_lcfs = 0.999;
  Vec3 b;
  double axis_psi = 0.0;
  if(!equilibrium_values(fields, axis_r, phi, axis_z, b, &axis_psi))
    return false;
  const double target_psi = axis_psi + inside_lcfs * (psi_lcfs - axis_psi);

  const double scale = std::max(1.0, std::abs(axis_r));
  const double step = 0.002 * scale;
  double previous_r = axis_r;
  double previous_value = axis_psi - target_psi;
  for(int i = 1; i <= 2000; ++i) {
    const double r = axis_r + i * step;
    double psi = 0.0;
    if(!equilibrium_values(fields, r, phi, axis_z, b, &psi)) {
      double lower = previous_r;
      double upper = r;
      for(int iteration = 0; iteration < 60; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if(equilibrium_values(fields, middle, phi, axis_z, b, &psi))
          lower = middle;
        else
          upper = middle;
      }
      *edge_r = lower;
      return lower > axis_r;
    }
    const double value = psi - target_psi;
    if(previous_value * value <= 0.0) {
      double lower = previous_r;
      double upper = r;
      for(int iteration = 0; iteration < 60; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if(!equilibrium_values(fields, middle, phi, axis_z, b, &psi))
          return false;
        if(previous_value * (psi - target_psi) <= 0.0)
          upper = middle;
        else
          lower = middle;
      }
      *edge_r = 0.5 * (lower + upper);
      return true;
    }
    previous_r = r;
    previous_value = value;
  }
  return false;
}

bool pphi_at_lfs_point(Fields& fields, double r, double phi, double z,
                       double energy, double mu, double mass, double charge,
                       int sigma, double* pphi, double* vparallel)
{
  Vec3 b;
  double psi = 0.0;
  if(!equilibrium_values(fields, r, phi, z, b, &psi)) return false;
  const double bmag = norm(b);
  const double parallel_energy = energy - mu * bmag;
  if(parallel_energy < 0.0) return false;

  *vparallel = sigma * std::sqrt(2.0 * parallel_energy / mass);
  const double i_over_b = r * b[1] / bmag;
  *pphi = psi + (mass / charge) * (*vparallel) * i_over_b;
  return true;
}

bool find_lfs_pphi_point(Fields& fields, double target_pphi, double energy,
                         double mu, double mass, double charge, int sigma,
                         double axis_r, double axis_z, double phi,
                         double edge_r, State& state)
{
  constexpr int samples = 2048;
  const double margin = 1.0e-8 * std::max(1.0, edge_r - axis_r);
  const double lower_r = axis_r + margin;
  const double upper_r = edge_r - margin;
  bool have_previous = false;
  bool found = false;
  double previous_r = 0.0;
  double previous_value = 0.0;
  double root_r = 0.0;
  double root_vparallel = 0.0;

  for(int i = 0; i <= samples; ++i) {
    const double r = lower_r + (upper_r - lower_r) * i / samples;
    double value = 0.0;
    double vparallel = 0.0;
    const bool valid = pphi_at_lfs_point(fields, r, phi, axis_z, energy, mu,
                                         mass, charge, sigma, &value,
                                         &vparallel);
    if(!valid) {
      have_previous = false;
      continue;
    }
    value -= target_pphi;
    if(have_previous && previous_value * value <= 0.0) {
      double lower = previous_r;
      double upper = r;
      double lower_value = previous_value;
      for(int iteration = 0; iteration < 60; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        double middle_pphi = 0.0;
        double middle_vparallel = 0.0;
        if(!pphi_at_lfs_point(fields, middle, phi, axis_z, energy, mu, mass,
                              charge, sigma, &middle_pphi,
                              &middle_vparallel))
          break;
        const double middle_value = middle_pphi - target_pphi;
        if(lower_value * middle_value <= 0.0) {
          upper = middle;
        } else {
          lower = middle;
          lower_value = middle_value;
        }
      }
      root_r = 0.5 * (lower + upper);
      double solved_pphi = 0.0;
      if(pphi_at_lfs_point(fields, root_r, phi, axis_z, energy, mu, mass,
                           charge, sigma, &solved_pphi, &root_vparallel))
        found = true;
    }
    previous_r = r;
    previous_value = value;
    have_previous = true;
  }

  if(!found) return false;
  state = {root_r, phi, axis_z, root_vparallel};
  return true;
}

bool rhs(Fields& fields, const State& state, double mu_over_q, double q_over_m,
         State& deriv)
{
  if(state[0] <= 0.0) return false;

  Vec3 b;
  std::array<double, 9> db;
  if(!magnetic_data(fields, state, b, db)) return false;

  const double bmag = norm(b);
  if(!std::isfinite(bmag) || bmag <= 0.0) return false;
  const Vec3 bhat = {b[0] / bmag, b[1] / bmag, b[2] / bmag};

  const Vec3 dbdr = {db[FIO_DR_R], db[FIO_DR_PHI], db[FIO_DR_Z]};
  const Vec3 dbdphi = {db[FIO_DPHI_R], db[FIO_DPHI_PHI], db[FIO_DPHI_Z]};
  const Vec3 dbdz = {db[FIO_DZ_R], db[FIO_DZ_PHI], db[FIO_DZ_Z]};
  const double rinv = 1.0 / state[0];
  const Vec3 grad_b = {dot(bhat, dbdr), rinv * dot(bhat, dbdphi), dot(bhat, dbdz)};

  // This is curl(B) in cylindrical physical components, named Jcyl in particle.f90.
  const Vec3 curl_b = {rinv * dbdphi[2] - dbdz[1],
                       dbdz[0] - dbdr[2],
                       dbdr[1] - rinv * dbdphi[0] + rinv * b[1]};
  const Vec3 b_cross_grad_b = cross(b, grad_b);
  Vec3 curl_bhat;
  Vec3 bstar;
  for(int i = 0; i < 3; ++i) {
    curl_bhat[i] = (curl_b[i] + b_cross_grad_b[i] / bmag) / bmag;
    bstar[i] = b[i] + state[3] * curl_bhat[i] / q_over_m;
  }
  const double bstar_parallel = dot(bstar, bhat);
  if(!std::isfinite(bstar_parallel)
     || std::abs(bstar_parallel) < 1.0e-12 * bmag)
    return false;

  Vec3 force_per_charge;
  for(int i = 0; i < 3; ++i)
    force_per_charge[i] = mu_over_q * grad_b[i];
  const Vec3 drift = cross(bhat, force_per_charge);

  Vec3 velocity;
  for(int i = 0; i < 3; ++i)
    velocity[i] = (state[3] * bstar[i] + drift[i]) / bstar_parallel;

  deriv[0] = velocity[0];
  deriv[1] = velocity[1] * rinv;
  deriv[2] = velocity[2];
  deriv[3] = -q_over_m * dot(bstar, force_per_charge) / bstar_parallel;
  return true;
}

State shifted(const State& state, const State& deriv, double factor)
{
  State result;
  for(int i = 0; i < 4; ++i) result[i] = state[i] + factor * deriv[i];
  return result;
}

bool rk4_step(Fields& fields, State& state, double dt, double mu_over_q,
              double q_over_m)
{
  State k1, k2, k3, k4;
  if(!rhs(fields, state, mu_over_q, q_over_m, k1)) return false;
  const State s2 = shifted(state, k1, 0.5 * dt);
  if(!rhs(fields, s2, mu_over_q, q_over_m, k2)) return false;
  const State s3 = shifted(state, k2, 0.5 * dt);
  if(!rhs(fields, s3, mu_over_q, q_over_m, k3)) return false;
  const State s4 = shifted(state, k3, dt);
  if(!rhs(fields, s4, mu_over_q, q_over_m, k4)) return false;
  for(int i = 0; i < 4; ++i)
    state[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
  return true;
}

bool write_state(std::ostream& out, Fields& fields, const State& state,
                 double time, double mass, double mu)
{
  Vec3 b;
  std::array<double, 9> db;
  if(!magnetic_data(fields, state, b, db)) return false;
  const double bmag = norm(b);
  const double vperp = std::sqrt(std::max(0.0, 2.0 * mu * bmag / mass));
  const double energy_kev = (0.5 * mass * state[3] * state[3] + mu * bmag)
                            / kev_to_joule;
  out << time << ' ' << state[0] << ' ' << state[1] << ' '
      << state[1] * 180.0 / pi << ' ' << state[2] << ' ' << state[3] << ' '
      << vperp << ' ' << energy_kev << ' ' << mu / kev_to_joule << ' '
      << bmag << '\n';
  return true;
}

std::vector<double> scan_values(const ScanRange& range, double scalar)
{
  if(!range.enabled()) return std::vector<double>(1, scalar);

  std::vector<double> values(range.count, range.minimum);
  if(range.count == 1) return values;
  const double spacing = (range.maximum - range.minimum) / (range.count - 1);
  for(int i = 0; i < range.count; ++i)
    values[i] = range.minimum + i * spacing;
  values.back() = range.maximum;
  return values;
}

const char* scan_status_name(int status)
{
  switch(status) {
  case scan_complete:          return "complete";
  case scan_incomplete:        return "incomplete";
  case scan_invalid_energy:    return "invalid_E_lt_muB";
  case scan_lost_or_singular:  return "lost_or_singular";
  case scan_invalid_pphi:      return "invalid_Pphi";
  case scan_output_error:      return "output_error";
  default:                     return "unknown";
  }
}

std::vector<ScanResult> gather_scan_results(const std::vector<ScanResult>& local,
                                            int rank, int size)
{
  const int local_count = static_cast<int>(local.size());
  std::vector<int> counts(rank == 0 ? size : 0);
  MPI_Gather(&local_count, 1, MPI_INT,
             rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> displacements(rank == 0 ? size : 0);
  int total_count = 0;
  if(rank == 0) {
    for(int i = 0; i < size; ++i) {
      displacements[i] = total_count;
      total_count += counts[i];
    }
  }

  std::vector<int> local_index(local_count);
  std::vector<int> local_steps(local_count);
  std::vector<int> local_status(local_count);
  std::vector<int> local_sigma(local_count);
  std::vector<int> local_orbit_type(local_count);
  std::vector<double> local_energy(local_count);
  std::vector<double> local_mu(local_count);
  std::vector<double> local_lambda(local_count);
  std::vector<double> local_pphi(local_count);
  std::vector<double> local_initial_r(local_count);
  std::vector<double> local_toroidal(local_count);
  std::vector<double> local_poloidal(local_count);
  std::vector<double> local_orbit_period(local_count);
  std::vector<double> local_jacobian(local_count);
  for(int i = 0; i < local_count; ++i) {
    local_index[i] = local[i].index;
    local_steps[i] = local[i].steps;
    local_status[i] = local[i].status;
    local_sigma[i] = local[i].sigma;
    local_orbit_type[i] = local[i].orbit_type;
    local_energy[i] = local[i].energy;
    local_mu[i] = local[i].mu;
    local_lambda[i] = local[i].lambda;
    local_pphi[i] = local[i].pphi;
    local_initial_r[i] = local[i].initial_r;
    local_toroidal[i] = local[i].toroidal_period;
    local_poloidal[i] = local[i].poloidal_period;
    local_orbit_period[i] = local[i].orbit_period;
    local_jacobian[i] = local[i].jacobian_relative;
  }

  std::vector<int> all_index(rank == 0 ? total_count : 0);
  std::vector<int> all_steps(rank == 0 ? total_count : 0);
  std::vector<int> all_status(rank == 0 ? total_count : 0);
  std::vector<int> all_sigma(rank == 0 ? total_count : 0);
  std::vector<int> all_orbit_type(rank == 0 ? total_count : 0);
  std::vector<double> all_energy(rank == 0 ? total_count : 0);
  std::vector<double> all_mu(rank == 0 ? total_count : 0);
  std::vector<double> all_lambda(rank == 0 ? total_count : 0);
  std::vector<double> all_pphi(rank == 0 ? total_count : 0);
  std::vector<double> all_initial_r(rank == 0 ? total_count : 0);
  std::vector<double> all_toroidal(rank == 0 ? total_count : 0);
  std::vector<double> all_poloidal(rank == 0 ? total_count : 0);
  std::vector<double> all_orbit_period(rank == 0 ? total_count : 0);
  std::vector<double> all_jacobian(rank == 0 ? total_count : 0);

#define GATHER_SCAN_VALUES(send, receive, type) \
  MPI_Gatherv((send).data(), local_count, type, \
              rank == 0 ? (receive).data() : nullptr, \
              rank == 0 ? counts.data() : nullptr, \
              rank == 0 ? displacements.data() : nullptr, type, 0, \
              MPI_COMM_WORLD)
  GATHER_SCAN_VALUES(local_index, all_index, MPI_INT);
  GATHER_SCAN_VALUES(local_steps, all_steps, MPI_INT);
  GATHER_SCAN_VALUES(local_status, all_status, MPI_INT);
  GATHER_SCAN_VALUES(local_sigma, all_sigma, MPI_INT);
  GATHER_SCAN_VALUES(local_orbit_type, all_orbit_type, MPI_INT);
  GATHER_SCAN_VALUES(local_energy, all_energy, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_mu, all_mu, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_lambda, all_lambda, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_pphi, all_pphi, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_initial_r, all_initial_r, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_toroidal, all_toroidal, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_poloidal, all_poloidal, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_orbit_period, all_orbit_period, MPI_DOUBLE);
  GATHER_SCAN_VALUES(local_jacobian, all_jacobian, MPI_DOUBLE);
#undef GATHER_SCAN_VALUES

  std::vector<ScanResult> ordered(rank == 0 ? total_count : 0);
  if(rank == 0) {
    for(int i = 0; i < total_count; ++i) {
      ScanResult& result = ordered[all_index[i]];
      result.index = all_index[i];
      result.steps = all_steps[i];
      result.status = all_status[i];
      result.sigma = all_sigma[i];
      result.orbit_type = all_orbit_type[i];
      result.energy = all_energy[i];
      result.mu = all_mu[i];
      result.lambda = all_lambda[i];
      result.pphi = all_pphi[i];
      result.initial_r = all_initial_r[i];
      result.toroidal_period = all_toroidal[i];
      result.poloidal_period = all_poloidal[i];
      result.orbit_period = all_orbit_period[i];
      result.jacobian_relative = all_jacobian[i];
    }
  }
  return ordered;
}

int scan_particle_periods(Fields& fields, const Parameters& p,
                          const State& initial_position, double axis_r,
                          double axis_z, double psi_lcfs, double reference_b,
                          double mass, double charge, int rank, int size)
{
  const bool pphi_scan = p.pphi_range.enabled() || std::isfinite(p.pphi);
  const bool lambda_input = p.lambda_range.enabled() || std::isfinite(p.lambda);
  Vec3 initial_b;
  std::array<double, 9> initial_db;
  double edge_r = 0.0;
  const bool local_field_ok = pphi_scan
    ? find_lfs_edge(fields, axis_r, axis_z, p.phi, psi_lcfs, &edge_r)
    : magnetic_data(fields, initial_position, initial_b, initial_db);
  const int local_field_error = local_field_ok ? 0 : 1;
  int field_error = 0;
  MPI_Allreduce(&local_field_error, &field_error, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if(field_error) {
    if(rank == 0)
      std::cerr << (pphi_scan
        ? "Could not locate the low-field-side LCFS\n"
        : "Initial position is outside the magnetic-field mesh\n");
    return 1;
  }
  const double initial_bmag = pphi_scan ? 0.0 : norm(initial_b);
  const double q_over_m = charge / mass;
  const std::vector<double> energies = scan_values(p.energy_range, p.energy_kev);
  const std::vector<double> pitch_values = lambda_input
    ? scan_values(p.lambda_range, p.lambda)
    : scan_values(p.mu_range, p.mu_kev_per_t);
  const std::vector<double> pphis = pphi_scan
    ? scan_values(p.pphi_range, p.pphi)
    : std::vector<double>(1, std::numeric_limits<double>::quiet_NaN());
  const std::vector<int> sigmas = p.sigma == 0
    ? std::vector<int>{1, -1} : std::vector<int>{p.sigma};
  const int energy_count = static_cast<int>(energies.size());
  const int pitch_count = static_cast<int>(pitch_values.size());
  const int sigma_count = static_cast<int>(sigmas.size());
  const int scan_count = static_cast<int>(pphis.size()) * energy_count
                         * pitch_count * sigma_count;
  std::vector<ScanResult> local_results;
  local_results.reserve((scan_count + size - 1 - rank) / size);

  for(int index = rank; index < scan_count; index += size) {
    const int pphi_index = index / (energy_count * pitch_count * sigma_count);
    const int remainder = index % (energy_count * pitch_count * sigma_count);
    const int energy_index = remainder / (pitch_count * sigma_count);
    const int pitch_sigma_index = remainder % (pitch_count * sigma_count);
    const int pitch_index = pitch_sigma_index / sigma_count;
    const int sigma = sigmas[pitch_sigma_index % sigma_count];
    const double target_pphi = pphis[pphi_index];
    const double energy_kev = energies[energy_index];
    const double lambda = lambda_input
      ? pitch_values[pitch_index]
      : pitch_values[pitch_index] * reference_b / energy_kev;
    const double mu_kev_per_t = lambda_input
      ? lambda * energy_kev / reference_b
      : pitch_values[pitch_index];
    const double mu = mu_kev_per_t * kev_to_joule;
    const double parallel_energy = energy_kev * kev_to_joule
                                   - mu * initial_bmag;
    ScanResult result;
    result.index = index;
    result.energy = energy_kev;
    result.mu = mu_kev_per_t;
    result.lambda = lambda;
    result.pphi = target_pphi;
    result.sigma = sigma;

    State state = initial_position;
    if(pphi_scan
       && !find_lfs_pphi_point(fields, target_pphi,
                               energy_kev * kev_to_joule, mu, mass, charge,
                               sigma, axis_r, axis_z, p.phi, edge_r, state)) {
      result.status = scan_invalid_pphi;
    } else if(!pphi_scan && parallel_energy < 0.0) {
      result.status = scan_invalid_energy;
    } else {
      if(!pphi_scan)
        state[3] = sigma * std::sqrt(2.0 * parallel_energy / mass);
      result.initial_r = state[0];
      MotionPeriodTracker tracker(axis_r, axis_z);
      initialize_poloidal_period(state, 0.0, tracker);

      std::ofstream poincare;
      if(p.pout) {
        const std::string filename = "out" + std::to_string(index);
        poincare.open(filename.c_str(), std::ios::out | std::ios::trunc);
        if(!poincare) {
          std::cerr << "Rank " << rank << " cannot open " << filename << '\n';
          result.status = scan_output_error;
        } else {
          poincare << "# phi_deg R_m Z_m theta_deg time_s\n"
                   << std::setprecision(16) << std::scientific;
        }
      }

      const double mu_over_q = mu / charge;
      const double initial_phi = state[1];
      for(int step = 1; step <= p.steps; ++step) {
        const State previous_state = state;
        const double previous_time = (step - 1) * p.dt;
        if(!rk4_step(fields, state, p.dt, mu_over_q, q_over_m)) {
          result.status = scan_lost_or_singular;
          break;
        }
        result.steps = step;
        calculate_motion_period(previous_state, state, previous_time,
                                step * p.dt, tracker);
        if(poincare)
          write_poincare_crossings(poincare, previous_state, state,
                                   previous_time, step * p.dt,
                                   p.poincare_plane, p.nplanes, axis_r, axis_z);
        const bool transit_target_reached = p.transits > 0
          && std::abs(state[1] - initial_phi) >= 2.0 * pi * p.transits;
        if(transit_target_reached)
          break;
        if(p.transits == 0 && p.qout && !p.pout
           && std::isfinite(tracker.periods.toroidal)
           && std::isfinite(tracker.periods.poloidal))
          break;
      }
      result.toroidal_period = tracker.periods.toroidal;
      result.poloidal_period = tracker.periods.poloidal;
      result.orbit_type = detected_orbit_type(tracker);
      result.orbit_period = tracker.periods.poloidal;
      result.jacobian_relative = relative_jacobian(
        result.orbit_period, energy_kev, reference_b
      );
      if(p.qout && result.status == scan_complete
         && (!std::isfinite(result.toroidal_period)
             || !std::isfinite(result.poloidal_period)))
        result.status = scan_incomplete;
    }
    local_results.push_back(result);
  }

  const std::vector<ScanResult> results = gather_scan_results(local_results,
                                                              rank, size);
  if(rank == 0 && p.qout) {
    std::ofstream out(p.output.c_str(), std::ios::out | std::ios::trunc);
    if(!out) {
      std::cerr << "Cannot open output file " << p.output << '\n';
      return 1;
    }
    if(pphi_scan) out << "# pphi_over_q_Wb initial_R_m ";
    out << "energy_keV mu_keV_per_T lambda sigma orbit_type "
           "toroidal_period_s poloidal_period_s orbit_period_s "
           "jacobian_relative_keV_s_per_T steps status\n";
    out << std::setprecision(16) << std::scientific;
    for(const ScanResult& result : results) {
      if(pphi_scan) out << result.pphi << ' ' << result.initial_r << ' ';
      out << result.energy << ' ' << result.mu << ' ' << result.lambda << ' '
          << result.sigma << ' ' << orbit_type_name(result.orbit_type) << ' '
          << result.toroidal_period << ' ' << result.poloidal_period << ' '
          << result.orbit_period << ' ' << result.jacobian_relative << ' '
          << result.steps << ' ' << scan_status_name(result.status) << '\n';
      if(pphi_scan) std::cout << "Pphi/q=" << result.pphi << " Wb, ";
      std::cout << "E=" << result.energy << " keV, mu=" << result.mu
                << " keV/T, Lambda=" << result.lambda
                << ", sigma=" << result.sigma
                << ", orbit=" << orbit_type_name(result.orbit_type)
                << ": T_tor=" << result.toroidal_period
                << " s, T_pol=" << result.poloidal_period
                << " s, T_orbit=" << result.orbit_period
                << " s, J_rel=" << result.jacobian_relative << " ("
                << scan_status_name(result.status) << ")\n";
    }
    std::cout << "Wrote period scan to " << p.output << '\n';
  }
  if(rank == 0 && p.pout)
    std::cout << "Wrote Poincare crossings to out<scan_index>\n";
  return 0;
}

int run(const Parameters& p, int rank, int size)
{
  Fields fields;
  const bool scan = p.energy_range.enabled() || p.mu_range.enabled()
                    || p.lambda_range.enabled()
                    || p.pphi_range.enabled() || std::isfinite(p.pphi)
                    || p.sigma == 0;
  const int local_load_error = load_fields(p, fields) ? 0 : 1;
  if(scan) {
    int load_error = 0;
    MPI_Allreduce(&local_load_error, &load_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if(load_error) return 1;
  } else if(local_load_error) {
    return 1;
  }

  double axis_r = 0.0;
  double axis_z = 0.0;
  double psi_lcfs = std::numeric_limits<double>::quiet_NaN();
  double slice_time = 0.0;
  fio_series* series = nullptr;
  if(fields.source.get_slice_time(p.timeslice, &slice_time) != FIO_SUCCESS)
    slice_time = 0.0;
  if(fields.source.get_series(FIO_MAGAXIS_R, &series) == FIO_SUCCESS) {
    series->eval(slice_time, &axis_r);
    delete series;
    series = nullptr;
  }
  if(fields.source.get_series(FIO_MAGAXIS_Z, &series) == FIO_SUCCESS) {
    series->eval(slice_time, &axis_z);
    delete series;
    series = nullptr;
  }
  if(fields.source.get_series(FIO_LCFS_PSI, &series) == FIO_SUCCESS) {
    series->eval(slice_time, &psi_lcfs);
    delete series;
  }

  State state = {std::isfinite(p.r) ? p.r : axis_r,
                 p.phi,
                 std::isfinite(p.z) ? p.z : axis_z,
                 0.0};
  const double mass_ratio = std::isfinite(p.mass_ratio) ? p.mass_ratio
                                                       : fields.source.ion_mass;
  if(mass_ratio <= 0.0) {
    std::cerr << "Particle mass must be positive\n";
    return 1;
  }
  const double mass = mass_ratio * proton_mass;
  const double charge_state = std::isfinite(p.charge_state) ? p.charge_state
                                                            : fields.source.z_ion;
  if(charge_state == 0.0) {
    std::cerr << "Particle charge must be nonzero\n";
    return 1;
  }
  const double charge = charge_state * elementary_charge;
  const double q_over_m = charge / mass;
  const double reference_b = std::abs(fields.source.bzero) * fields.source.B0;
  const bool lambda_input = p.lambda_range.enabled() || std::isfinite(p.lambda);
  if(lambda_input
     && (!std::isfinite(reference_b) || reference_b <= 0.0)) {
    std::cerr << "A finite positive |bzero|*b0_norm is required for --lambda\n";
    return 1;
  }

  if(scan)
    return scan_particle_periods(fields, p, state, axis_r, axis_z, psi_lcfs,
                                 reference_b, mass, charge, rank, size);

  const double mu_kev_per_t = lambda_input
    ? p.lambda * p.energy_kev / reference_b : p.mu_kev_per_t;
  const double lambda = lambda_input
    ? p.lambda : mu_kev_per_t * reference_b / p.energy_kev;
  const double mu = mu_kev_per_t * kev_to_joule;
  const double mu_over_q = mu / charge;

  Vec3 initial_b;
  std::array<double, 9> initial_db;
  if(!magnetic_data(fields, state, initial_b, initial_db)) {
    std::cerr << "Initial position is outside the magnetic-field mesh\n";
    return 1;
  }
  const double parallel_energy = p.energy_kev * kev_to_joule - mu * norm(initial_b);
  if(parallel_energy < 0.0) {
    std::cerr << "Initial energy is below mu*B: E=" << p.energy_kev
              << " keV, mu*B=" << mu * norm(initial_b) / kev_to_joule
              << " keV\n";
    return 1;
  }
  state[3] = p.sigma * std::sqrt(2.0 * parallel_energy / mass);

  std::ofstream out(p.output.c_str(), std::ios::out | std::ios::trunc);
  if(!out) {
    std::cerr << "Cannot open output file " << p.output << '\n';
    return 1;
  }
  out << "# t_s R_m phi_rad phi_deg Z_m vpar_m_per_s vperp_m_per_s "
         "energy_keV mu_keV_per_T B_T\n";
  out << std::setprecision(16) << std::scientific;
  if(!write_state(out, fields, state, 0.0, mass, mu)) {
    std::cerr << "Could not evaluate the initial output state\n";
    return 1;
  }

  MotionPeriodTracker period_tracker(axis_r, axis_z);
  initialize_poloidal_period(state, 0.0, period_tracker);

  std::ofstream poincare;
  if(p.pout) {
    poincare.open("out0", std::ios::out | std::ios::trunc);
    if(!poincare) {
      std::cerr << "Cannot open out0\n";
      return 1;
    }
    poincare << "# phi_deg R_m Z_m theta_deg time_s\n"
             << std::setprecision(16) << std::scientific;
  }

  int completed = 0;
  bool transit_target_reached = false;
  const double initial_phi = state[1];
  for(int step = 1; step <= p.steps; ++step) {
    const State previous_state = state;
    const double previous_time = (step - 1) * p.dt;
    if(!rk4_step(fields, state, p.dt, mu_over_q, q_over_m)) {
      std::cerr << "Orbit left the field mesh or encountered a singular B*_parallel at step "
                << step << '\n';
      break;
    }
    completed = step;
    calculate_motion_period(previous_state, state, previous_time, step * p.dt,
                            period_tracker);
    if(poincare)
      write_poincare_crossings(poincare, previous_state, state, previous_time,
                               step * p.dt, p.poincare_plane, p.nplanes,
                               axis_r, axis_z);
    transit_target_reached = p.transits > 0
      && std::abs(state[1] - initial_phi) >= 2.0 * pi * p.transits;
    if((step % p.output_every == 0 || step == p.steps
        || transit_target_reached)
       && !write_state(out, fields, state, step * p.dt, mass, mu)) {
      std::cerr << "Could not evaluate output at step " << step << '\n';
      return 2;
    }
    if(transit_target_reached)
      break;
  }

  if(p.qout) {
    if(std::isfinite(period_tracker.periods.toroidal)) {
      out << "# toroidal_period_s " << period_tracker.periods.toroidal << '\n';
      std::cout << "Toroidal motion period: " << period_tracker.periods.toroidal
                << " s\n";
    } else {
      out << "# toroidal_period_s nan\n";
      std::cout << "Toroidal motion period was not completed\n";
    }
    if(std::isfinite(period_tracker.periods.poloidal)) {
      out << "# poloidal_period_s " << period_tracker.periods.poloidal << '\n';
      std::cout << "Poloidal motion period: " << period_tracker.periods.poloidal
                << " s\n";
    } else {
      out << "# poloidal_period_s nan\n";
      std::cout << "Poloidal motion period was not completed\n";
    }
    const int orbit_type = detected_orbit_type(period_tracker);
    const double orbit_period = period_tracker.periods.poloidal;
    const double jacobian = relative_jacobian(
      orbit_period, p.energy_kev, reference_b
    );
    out << "# lambda " << lambda << '\n';
    out << "# orbit_type " << orbit_type_name(orbit_type) << '\n';
    out << "# orbit_period_s " << orbit_period << '\n';
    out << "# jacobian_relative_keV_s_per_T " << jacobian << '\n';
    std::cout << "Orbit type: " << orbit_type_name(orbit_type) << '\n';
    std::cout << "Orbit period: " << orbit_period << " s\n";
    std::cout << "Relative COM Jacobian: " << jacobian << " keV s/T\n";
  }
  if(p.pout) std::cout << "Wrote Poincare crossings to out0\n";

  std::cout << "Wrote " << p.output << " after " << completed << " of "
            << p.steps << " steps\n";
  return completed == p.steps || transit_target_reached ? 0 : 2;
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int status = 0;
  try {
    Parameters parameters;
    if(parse_command_line(argc, argv, parameters, rank == 0)) {
      const bool scan = parameters.energy_range.enabled()
                        || parameters.mu_range.enabled()
                        || parameters.pphi_range.enabled()
                        || std::isfinite(parameters.pphi);
      if(scan || rank == 0) status = run(parameters, rank, size);
    }
  } catch(const std::exception& error) {
    if(rank == 0) std::cerr << "Error: " << error.what() << '\n';
    status = 1;
  }

  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  MPI_Finalize();
  return global_status;
}
