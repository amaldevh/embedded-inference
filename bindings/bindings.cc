#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
namespace py=pybind11;
namespace std{
template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}
};

#include "waypoint_generator.hh"
#include "preprocessor.hh"
//#include <pybind11/native_enum.h>

PYBIND11_MODULE(bindings, m){
	using Point =  std::vector<float>;
	using  FeasibilityLimits = WaypointGenerator::FeasibilityLimits;
	using State = Eigen::Matrix<float, 186, 1>;
    using Statef = Eigen::Matrix<float, 13, 1>;
    using Action = Eigen::Matrix<float, 6, 1>;
	py::class_<FeasibilityLimits>(m, "FeasibilityLimits")
	.def(py::init<>())
	.def_readwrite("min_speed", &FeasibilityLimits::min_speed)
    .def_readwrite("max_speed", &FeasibilityLimits::max_speed)

    .def_readwrite("max_normal_acceleration", &FeasibilityLimits::max_normal_acceleration)

    .def_readwrite("gravity", &FeasibilityLimits::gravity)
    .def_readwrite("max_specific_thrust", &FeasibilityLimits::max_specific_thrust)
    .def_readwrite("max_tilt_rad", &FeasibilityLimits::max_tilt_rad)

    .def_readwrite("safety_factor", &FeasibilityLimits::safety_factor)

    .def_readwrite("max_tangent_change_rad", &FeasibilityLimits::max_tangent_change_rad)

    .def_readwrite("feasibility_samples", &FeasibilityLimits::feasibility_samples)
    .def_readwrite("integration_arc_step", &FeasibilityLimits::integration_arc_step)
    .def_readwrite("max_parameter_step", &FeasibilityLimits::max_parameter_step)
    .def_readwrite("min_parameter_speed", &FeasibilityLimits::min_parameter_speed)

    .def_readwrite("max_yaw_rate_rad_s", &FeasibilityLimits::max_yaw_rate_rad_s);
	py::class_<WaypointGenerator>(m, "WaypointGenerator")
	.def(py::init<>())
	.def("GenerateCircleWaypoints", static_cast< std::vector<Point> (WaypointGenerator::*) (float, float, float, float, float, float, const FeasibilityLimits&)>(&WaypointGenerator::GenerateCircleWaypoints), py::call_guard<py::gil_scoped_release>())
	.def("GenerateLissajousWaypoints", static_cast< std::vector<Point> (WaypointGenerator::*) (float, float, float, float, float, float, float, float, float, float, float, float,  const FeasibilityLimits&)>( &WaypointGenerator::GenerateLissajousWaypoints), py::call_guard<py::gil_scoped_release>())
	.def("GenerateLissajousWaypoints", static_cast< std::vector<Point> (WaypointGenerator::*) (float, float,  float, float,  float, float,  float, float, float, const FeasibilityLimits&)>( &WaypointGenerator::GenerateLissajousWaypoints), py::call_guard<py::gil_scoped_release>());
	py::class_<TrajectoryPlanningProcessor>(m, "TrajectoryPlanningProcessor")
	.def(py::init<int, int, bool, bool, const std::vector<std::vector<float>>&, float, float, float, const Eigen::VectorXf&, const Eigen::VectorXf&,  TrajectoryPlanningProcessor::GateAcceptanceMode>())
	.def("Reset", &TrajectoryPlanningProcessor::Reset, py::call_guard<py::gil_scoped_release>())
	.def("ProcessObservation", &TrajectoryPlanningProcessor::ProcessObservation,  py::call_guard<py::gil_scoped_release>());
	py::enum_<TrajectoryPlanningProcessor::GateAcceptanceMode>(m, "GateAcceptanceMode")
	.value("FullRectangularOpening", TrajectoryPlanningProcessor::GateAcceptanceMode::FullRectangularOpening)
	.value("CentralCircularCorridor", TrajectoryPlanningProcessor::GateAcceptanceMode::CentralCircularCorridor)
	.export_values();
}
