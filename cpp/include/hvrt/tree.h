#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include "hvrt/types.h"

namespace hvrt {

// ── Tree node ─────────────────────────────────────────────────────────────────

// Layout: doubles first, then ints, then bools — minimises padding on 64-bit.
struct TreeNode {
    double threshold     = 0.0;
    double leaf_value    = 0.0;    // mean target at leaf (used for GBT prediction)
    int    feature_idx   = -1;     // global feature index (in X_z)
    int    bin_threshold = -1;     // bin index for binned predict (go left if bin <= this)
    int    left          = -1;     // child node index
    int    right         = -1;
    int    partition_id  = -1;     // assigned at leaf (HVRT partition routing)
    bool   is_binary     = false;  // true → binary-stream split (value <= threshold)
    bool   is_leaf       = false;
};

// ── Partition tree ────────────────────────────────────────────────────────────

class PartitionTree {
public:
    Eigen::VectorXi build(
        const Eigen::MatrixXd& X_z,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
        const std::vector<int>& cont_cols,
        const std::vector<int>& binary_cols,
        const Eigen::VectorXd& target,
        const HVRTConfig& cfg,
        const Eigen::VectorXd& hessians = Eigen::VectorXd(),
        Eigen::VectorXd* train_preds = nullptr);

    Eigen::VectorXi apply(const Eigen::MatrixXd& X_z) const;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    void predict_into(const Eigen::MatrixXd& X, Eigen::VectorXd& out) const;

    void predict_binned_into(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_bin,
        Eigen::VectorXd& out) const;

    void predict_leaf_node_binned_into(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_bin,
        Eigen::VectorXi& out) const;

    const std::vector<double>& feature_importances() const { return feature_importances_; }
    int n_leaves() const { return n_leaves_; }
    bool fitted()  const { return fitted_; }

    const std::vector<int16_t>&  flat_feature()     const { return flat_feature_; }
    const std::vector<uint8_t>&  flat_bin_thresh()  const { return flat_bin_thresh_; }
    const std::vector<int32_t>&  flat_left()        const { return flat_left_; }
    const std::vector<int32_t>&  flat_right()       const { return flat_right_; }
    const std::vector<double>&   flat_leaf_value()  const { return flat_leaf_value_; }
    std::vector<double>&         flat_leaf_value_mut()    { return flat_leaf_value_; }
    int flat_n_nodes() const { return static_cast<int>(flat_feature_.size()); }

    void compile_lookup();
    bool has_lookup() const { return !lookup_table_.empty(); }

    void predict_lookup_into(
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_bin,
        Eigen::VectorXd& out) const;

    std::vector<uint8_t> to_bytes() const;
    void from_bytes(const std::vector<uint8_t>& data);

    void from_flat(
        std::vector<int16_t>  feature,
        std::vector<uint8_t>  bin_thresh,
        std::vector<int32_t>  left,
        std::vector<int32_t>  right,
        std::vector<double>   leaf_value,
        int n_leaves) {
        flat_feature_    = std::move(feature);
        flat_bin_thresh_ = std::move(bin_thresh);
        flat_left_       = std::move(left);
        flat_right_      = std::move(right);
        flat_leaf_value_ = std::move(leaf_value);
        n_leaves_        = n_leaves;
        flat_valid_      = true;
        fitted_          = true;
        const int nn = static_cast<int>(flat_feature_.size());
        nodes_.resize(nn);
        feature_importances_.clear();
        for (int i = 0; i < nn; ++i) {
            auto& nd = nodes_[i];
            nd.is_leaf       = (flat_feature_[i] < 0);
            nd.feature_idx   = nd.is_leaf ? -1 : flat_feature_[i];
            nd.bin_threshold = flat_bin_thresh_[i];
            nd.left          = flat_left_[i];
            nd.right         = flat_right_[i];
            nd.leaf_value    = flat_leaf_value_[i];
            nd.threshold     = 0.0;
            nd.is_binary     = false;
        }
    }

    void inject_bin_edges(const std::vector<Eigen::VectorXd>& edges,
                          const std::vector<int>& cont_cols,
                          int n_bins) {
        bin_edges_        = edges;
        cont_cols_cached_ = cont_cols;
        n_bins_cached_    = n_bins;
        bin_edges_valid_  = true;
    }

    static std::pair<int, int> auto_tune_params(int n, int d, bool for_reduction);

private:
    struct SplitResult {
        bool   valid      = false;
        int    feature    = -1;
        double threshold  = 0.0;
        bool   is_binary  = false;
        double gain       = 0.0;
        int    bin        = -1;
    };

    struct HistogramData {
        std::vector<double> bin_sum;
        std::vector<double> bin_sum_sq;
        std::vector<int>    bin_cnt;
        double sum_p = 0.0;
        double sum_sq_p = 0.0;
    };

    static SplitResult evaluate_continuous_splits(
        const std::vector<int>& indices,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
        const std::vector<int>& cont_cols,
        const Eigen::VectorXd& target,
        const std::vector<Eigen::VectorXd>& bin_edges,
        int n_bins,
        int min_samples_leaf,
        SplitStrategy strategy = SplitStrategy::Best,
        uint64_t rng_state = 0,
        const std::vector<double>* parent_sum = nullptr,
        const std::vector<double>* parent_sum_sq = nullptr,
        const std::vector<int>*    parent_cnt = nullptr,
        double parent_target_sum = 0.0,
        double parent_target_sum_sq = 0.0,
        const std::vector<int>* feature_subset = nullptr,
        HistogramData* hist_out = nullptr);

    SplitResult evaluate_binary_splits(
        const std::vector<int>& indices,
        const Eigen::MatrixXd& X_z,
        const std::vector<int>& binary_cols,
        const Eigen::VectorXd& target) const;

    std::vector<TreeNode> nodes_;
    std::vector<double>   feature_importances_;
    int d_full_  = 0;
    int n_leaves_= 0;
    bool fitted_ = false;

    std::vector<int16_t>  flat_feature_;
    std::vector<uint8_t>  flat_bin_thresh_;
    std::vector<int32_t>  flat_left_;
    std::vector<int32_t>  flat_right_;
    std::vector<double>   flat_leaf_value_;
    bool flat_valid_ = false;

    void build_flat_layout();

    std::vector<Eigen::VectorXd> bin_edges_;
    std::vector<int> cont_cols_cached_;
    std::vector<int> binary_cols_cached_;
    int n_bins_cached_ = 32;
    bool bin_edges_valid_ = false;

    std::vector<double>  lookup_table_;
    std::vector<int16_t> lookup_feats_;
    std::vector<uint8_t> lookup_thresh_;
    int lookup_depth_ = 0;
};

} // namespace hvrt
