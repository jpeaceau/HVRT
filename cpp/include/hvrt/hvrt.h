#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <optional>
#include "hvrt/types.h"
#include "hvrt/whitener.h"
#include "hvrt/binner.h"
#include "hvrt/tree.h"
#include "hvrt/expand.h"

namespace hvrt {

class HVRT {
public:
    explicit HVRT(HVRTConfig cfg = HVRTConfig{});

    // ── Fit ───────────────────────────────────────────────────────────────────
    // X            : n x d  (raw, not whitened)
    // y            : optional target variable (used for y_weight blending)
    // feature_types: "continuous" / "categorical" per column; auto-detect if empty
    HVRT& fit(const Eigen::MatrixXd& X,
              std::optional<Eigen::VectorXd> y = std::nullopt,
              std::optional<std::vector<std::string>> feature_types = std::nullopt);

    // ── Reduce ────────────────────────────────────────────────────────────────
    // Returns selected rows from original X (un-whitened).
    Eigen::MatrixXd reduce(
        std::optional<int>    n       = std::nullopt,
        std::optional<double> ratio   = std::nullopt,
        const std::string&    method  = "centroid_fps",
        bool var_weighted             = true,
        std::optional<int>    n_parts = std::nullopt) const;

    // Return indices only
    std::vector<int> reduce_indices(
        std::optional<int>    n       = std::nullopt,
        std::optional<double> ratio   = std::nullopt,
        const std::string&    method  = "centroid_fps",
        bool var_weighted             = true,
        std::optional<int>    n_parts = std::nullopt) const;

    // ── Expand ────────────────────────────────────────────────────────────────
    // Returns n synthetic samples in the original (un-whitened) feature space.
    Eigen::MatrixXd expand(
        int n,
        bool var_weighted                        = true,
        std::optional<float> bandwidth           = std::nullopt,
        const std::string& strategy              = "auto",
        bool adaptive_bandwidth                  = false,
        std::optional<int> n_parts               = std::nullopt) const;

    // ── Augment ───────────────────────────────────────────────────────────────
    // Augment = expand n rows and concatenate with original X.
    Eigen::MatrixXd augment(
        int n,
        bool var_weighted    = true,
        std::optional<int> n_parts = std::nullopt) const;

    // ── Partitioning info ─────────────────────────────────────────────────────
    std::vector<PartitionInfo> get_partitions() const;

    // ── Novelty ───────────────────────────────────────────────────────────────
    // Minimum distance from each row of X_new to the training set.
    Eigen::VectorXd compute_novelty(const Eigen::MatrixXd& X_new) const;

    // ── Tree apply ────────────────────────────────────────────────────────────
    // Assign partition IDs to new data.
    Eigen::VectorXi apply(const Eigen::MatrixXd& X_new) const;

    // ── Static helpers ────────────────────────────────────────────────────────
    static ParamRecommendation recommend_params(const Eigen::MatrixXd& X);

    // ── Stored attributes (for Python bindings / GeoXGB) ─────────────────────
    const Eigen::MatrixXd&  X_z()                const { return X_z_; }
    const Eigen::VectorXi&  partition_ids()       const { return partition_ids_; }
    std::vector<int>         unique_partitions()  const;
    const Whitener&          whitener()           const { return whitener_; }
    const PartitionTree&     tree()               const { return tree_; }

    // _to_z: whiten new data (same as whitener_.transform)
    Eigen::MatrixXd to_z(const Eigen::MatrixXd& X) const;

    bool fitted() const { return fitted_; }

private:
    HVRTConfig cfg_;

    // Components
    Whitener     whitener_;
    Binner       binner_;
    PartitionTree tree_;
    Expander     expander_;

    // Stored fit state
    Eigen::MatrixXd  X_z_;          // whitened training data
    Eigen::MatrixXd  X_orig_;       // original training data
    Eigen::VectorXi  partition_ids_;
    std::vector<int> cont_cols_;
    std::vector<int> cat_cols_;
    bool fitted_ = false;

    // Helpers
    static ReductionMethod    parse_reduction_method(const std::string& s);
    static GenerationStrategy parse_generation_strategy(const std::string& s);

    std::vector<bool> detect_feature_types(
        const Eigen::MatrixXd& X,
        const std::optional<std::vector<std::string>>& feature_types) const;
};

} // namespace hvrt
