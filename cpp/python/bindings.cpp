#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hvrt/hvrt.h"
#include "hvrt/types.h"
#include "hvrt/tree.h"
#include "hvrt/target.h"
#include "hvrt/reduce.h"

namespace py = pybind11;
using namespace hvrt;

// ── Eigen ↔ NumPy zero-copy helpers ──────────────────────────────────────────
// pybind11/eigen.h handles this automatically via Ref<> and MatrixXd returns.

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_hvrt_cpp, m) {
    m.doc() = "HVRT C++ backend — Hierarchical Variance-Retaining Transformer";

    // ── HVRTConfig ────────────────────────────────────────────────────────────
    py::class_<HVRTConfig>(m, "HVRTConfig")
        .def(py::init<>())
        .def_readwrite("n_partitions",     &HVRTConfig::n_partitions)
        .def_readwrite("min_samples_leaf", &HVRTConfig::min_samples_leaf)
        .def_readwrite("max_depth",        &HVRTConfig::max_depth)
        .def_readwrite("n_bins",           &HVRTConfig::n_bins)
        .def_readwrite("n_threads",        &HVRTConfig::n_threads)
        .def_readwrite("random_state",     &HVRTConfig::random_state)
        .def_readwrite("y_weight",         &HVRTConfig::y_weight)
        .def_readwrite("auto_tune",        &HVRTConfig::auto_tune)
        .def_readwrite("bandwidth",        &HVRTConfig::bandwidth);

    // ── PartitionInfo ─────────────────────────────────────────────────────────
    py::class_<PartitionInfo>(m, "PartitionInfo")
        .def_readonly("id",          &PartitionInfo::id)
        .def_readonly("size",        &PartitionInfo::size)
        .def_readonly("mean_abs_z",  &PartitionInfo::mean_abs_z)
        .def_readonly("variance",    &PartitionInfo::variance);

    // ── ParamRecommendation ───────────────────────────────────────────────────
    py::class_<ParamRecommendation>(m, "ParamRecommendation")
        .def_readonly("n_partitions",     &ParamRecommendation::n_partitions)
        .def_readonly("min_samples_leaf", &ParamRecommendation::min_samples_leaf);

    // ── PartitionTree (exposed for hvrt.tree_.apply and feature_importances_) ─
    py::class_<PartitionTree>(m, "PartitionTree")
        .def("apply", [](const PartitionTree& t, const Eigen::MatrixXd& X) {
            py::gil_scoped_release release;
            return t.apply(X);
        }, py::arg("X"))
        .def_property_readonly("feature_importances_",
            [](const PartitionTree& t) -> Eigen::VectorXd {
                const auto& fi = t.feature_importances();
                return Eigen::Map<const Eigen::VectorXd>(fi.data(), fi.size());
            })
        .def_property_readonly("n_leaves", &PartitionTree::n_leaves);

    // ── HVRT ─────────────────────────────────────────────────────────────────
    py::class_<HVRT>(m, "HVRT")
        .def(py::init<HVRTConfig>(), py::arg("config") = HVRTConfig{})

        // ── fit ──────────────────────────────────────────────────────────────
        .def("fit",
            [](HVRT& self,
               const Eigen::MatrixXd& X,
               py::object y_obj,
               py::object ft_obj) -> HVRT& {
                std::optional<Eigen::VectorXd> y;
                if (!y_obj.is_none()) {
                    y = y_obj.cast<Eigen::VectorXd>();
                }
                std::optional<std::vector<std::string>> ft;
                if (!ft_obj.is_none()) {
                    ft = ft_obj.cast<std::vector<std::string>>();
                }
                py::gil_scoped_release release;
                return self.fit(X, y, ft);
            },
            py::arg("X"),
            py::arg("y")            = py::none(),
            py::arg("feature_types")= py::none(),
            py::return_value_policy::reference)

        // ── reduce ────────────────────────────────────────────────────────────
        .def("reduce",
            [](const HVRT& self,
               py::object n_obj,
               py::object ratio_obj,
               const std::string& method,
               bool var_weighted,
               py::object n_parts_obj) {
                std::optional<int> n;
                if (!n_obj.is_none()) n = n_obj.cast<int>();
                std::optional<double> ratio;
                if (!ratio_obj.is_none()) ratio = ratio_obj.cast<double>();
                std::optional<int> np;
                if (!n_parts_obj.is_none()) np = n_parts_obj.cast<int>();
                py::gil_scoped_release release;
                return self.reduce(n, ratio, method, var_weighted, np);
            },
            py::arg("n")             = py::none(),
            py::arg("ratio")         = py::none(),
            py::arg("method")        = "centroid_fps",
            py::arg("variance_weighted") = true,
            py::arg("n_parts")       = py::none())

        // ── reduce (return_indices variant) ───────────────────────────────────
        .def("reduce_indices",
            [](const HVRT& self,
               py::object n_obj,
               py::object ratio_obj,
               const std::string& method,
               bool var_weighted,
               py::object n_parts_obj) {
                std::optional<int> n;
                if (!n_obj.is_none()) n = n_obj.cast<int>();
                std::optional<double> ratio;
                if (!ratio_obj.is_none()) ratio = ratio_obj.cast<double>();
                std::optional<int> np;
                if (!n_parts_obj.is_none()) np = n_parts_obj.cast<int>();
                py::gil_scoped_release release;
                return self.reduce_indices(n, ratio, method, var_weighted, np);
            },
            py::arg("n")             = py::none(),
            py::arg("ratio")         = py::none(),
            py::arg("method")        = "centroid_fps",
            py::arg("variance_weighted") = true,
            py::arg("n_parts")       = py::none())

        // ── expand ────────────────────────────────────────────────────────────
        .def("expand",
            [](const HVRT& self,
               int n,
               bool var_weighted,
               py::object bw_obj,
               const std::string& strategy,
               bool adaptive_bw,
               py::object n_parts_obj) {
                std::optional<float> bw;
                if (!bw_obj.is_none()) bw = bw_obj.cast<float>();
                std::optional<int> np;
                if (!n_parts_obj.is_none()) np = n_parts_obj.cast<int>();
                py::gil_scoped_release release;
                return self.expand(n, var_weighted, bw, strategy, adaptive_bw, np);
            },
            py::arg("n"),
            py::arg("variance_weighted")   = true,
            py::arg("bandwidth")           = py::none(),
            py::arg("generation_strategy") = "auto",
            py::arg("adaptive_bandwidth")  = false,
            py::arg("n_parts")             = py::none())

        // ── augment ───────────────────────────────────────────────────────────
        .def("augment",
            [](const HVRT& self, int n, bool vw, py::object np_obj) {
                std::optional<int> np;
                if (!np_obj.is_none()) np = np_obj.cast<int>();
                py::gil_scoped_release release;
                return self.augment(n, vw, np);
            },
            py::arg("n"),
            py::arg("variance_weighted") = true,
            py::arg("n_parts")           = py::none())

        // ── get_partitions ────────────────────────────────────────────────────
        .def("get_partitions",
            [](const HVRT& self) {
                py::gil_scoped_release release;
                return self.get_partitions();
            })

        // ── compute_novelty ───────────────────────────────────────────────────
        .def("compute_novelty",
            [](const HVRT& self, const Eigen::MatrixXd& X_new) {
                py::gil_scoped_release release;
                return self.compute_novelty(X_new);
            },
            py::arg("X"))

        // ── apply ─────────────────────────────────────────────────────────────
        .def("apply",
            [](const HVRT& self, const Eigen::MatrixXd& X_new) {
                py::gil_scoped_release release;
                return self.apply(X_new);
            },
            py::arg("X"))

        // ── _to_z (internal — used by GeoXGB) ────────────────────────────────
        .def("_to_z",
            [](const HVRT& self, const Eigen::MatrixXd& X) {
                py::gil_scoped_release release;
                return self.to_z(X);
            },
            py::arg("X"))

        // ── Stored attributes (used by GeoXGB) ────────────────────────────────
        .def_property_readonly("X_z_",
            [](const HVRT& self) -> const Eigen::MatrixXd& { return self.X_z(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly("partition_ids_",
            [](const HVRT& self) -> const Eigen::VectorXi& { return self.partition_ids(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly("unique_partitions_",
            [](const HVRT& self) { return self.unique_partitions(); })
        .def_property_readonly("tree_",
            [](const HVRT& self) -> const PartitionTree& { return self.tree(); },
            py::return_value_policy::reference_internal)

        // ── Static helpers ────────────────────────────────────────────────────
        .def_static("recommend_params",
            [](const Eigen::MatrixXd& X) {
                py::gil_scoped_release release;
                return HVRT::recommend_params(X);
            },
            py::arg("X"))

        .def_property_readonly("fitted_", &HVRT::fitted);

    // ── Standalone hot-path functions for Python dispatch ────────────────────
    m.def("compute_pairwise_target",
        [](const Eigen::MatrixXd& X_z) {
            py::gil_scoped_release release;
            return hvrt::compute_pairwise_target(X_z);
        }, py::arg("X_z"),
        "Pairwise interaction target. O(n·d²). Returns z-scored n-vector.");

    m.def("centroid_fps",
        [](const Eigen::MatrixXd& X_part, int budget) {
            py::gil_scoped_release release;
            return hvrt::centroid_fps(X_part, budget);
        }, py::arg("X"), py::arg("budget"),
        "Centroid-seeded FPS. Returns vector of local indices.");

    m.def("medoid_fps",
        [](const Eigen::MatrixXd& X_part, int budget) {
            py::gil_scoped_release release;
            return hvrt::medoid_fps(X_part, budget);
        }, py::arg("X"), py::arg("budget"),
        "Medoid-seeded FPS. Returns vector of local indices.");
}
