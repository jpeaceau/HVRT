#include "hvrt/tree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace hvrt {

// ── Auto-tune ────────────────────────────────────────────────────────────────

std::pair<int,int> PartitionTree::auto_tune_params(int n, int d, bool for_reduction) {
    int msl, max_leaf;
    if (for_reduction) {
        msl = std::max(5, (d * 40 * 2) / 3);
    } else {
        msl = static_cast<int>(std::max(static_cast<double>(d + 2),
                                        std::sqrt(static_cast<double>(n))));
    }
    max_leaf = std::max(30, std::min(1500, 3 * n / (msl * 2)));
    return {max_leaf, msl};
}

// ── Variance reduction gain ───────────────────────────────────────────────────
// gain = n * var(parent) - n_left * var(left) - n_right * var(right)
// Using Welford / running formula:
//   var_gain = (sum_sq - sum*sum/n) - [(sum_sq_L - sum_L^2/n_L) + (sum_sq_R - sum_R^2/n_R)]
// Equivalent to: (sum_L/n_L - sum_R/n_R)^2 * n_L*n_R / n  (for equal-variance split gain)

static double variance_gain(double sum_p, double sum_sq_p, int n_p,
                             double sum_l, double sum_sq_l, int n_l) {
    if (n_l <= 0 || n_l >= n_p) return 0.0;
    int n_r = n_p - n_l;
    double sum_r = sum_p - sum_l;
    double sum_sq_r = sum_sq_p - sum_sq_l;

    // variance of parent node (unnormalised: sum_sq - sum^2/n)
    double var_p   = sum_sq_p - sum_p  * sum_p  / n_p;
    double var_l   = sum_sq_l - sum_l  * sum_l  / n_l;
    double var_r   = sum_sq_r - sum_r  * sum_r  / n_r;

    double gain = var_p - var_l - var_r;
    return gain;
}

// ── Continuous split evaluation ───────────────────────────────────────────────

PartitionTree::SplitResult PartitionTree::evaluate_continuous_splits(
    const std::vector<int>& indices,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const Eigen::VectorXd& target,
    const std::vector<Eigen::VectorXd>& bin_edges,
    int n_bins) const
{
    const int n_node = static_cast<int>(indices.size());
    const int d_cont = static_cast<int>(cont_cols.size());

    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    // Pre-compute parent sum and sum_sq
    double sum_p = 0.0, sum_sq_p = 0.0;
    for (int idx : indices) {
        double t = target[idx];
        sum_p   += t;
        sum_sq_p+= t * t;
    }

    for (int fi = 0; fi < d_cont; ++fi) {
        int nb = static_cast<int>(bin_edges[fi].size()) - 1;
        if (nb <= 0) continue;

        // Stage A: scatter → per-bin sum and count
        std::vector<double> bin_sum(nb, 0.0);
        std::vector<double> bin_sum_sq(nb, 0.0);
        std::vector<int>    bin_cnt(nb, 0);

        for (int idx : indices) {
            uint8_t b = X_binned(idx, fi);
            if (b >= static_cast<uint8_t>(nb)) b = static_cast<uint8_t>(nb - 1);
            bin_sum[b]    += target[idx];
            bin_sum_sq[b] += target[idx] * target[idx];
            bin_cnt[b]    += 1;
        }

        // Stage B: prefix sums → scan thresholds
        double cum_sum = 0.0, cum_sum_sq = 0.0;
        int    cum_cnt = 0;
        for (int b = 0; b < nb - 1; ++b) {
            cum_sum    += bin_sum[b];
            cum_sum_sq += bin_sum_sq[b];
            cum_cnt    += bin_cnt[b];

            if (cum_cnt == 0 || cum_cnt == n_node) continue;

            double g = variance_gain(sum_p, sum_sq_p, n_node,
                                     cum_sum, cum_sum_sq, cum_cnt);
            if (g > best.gain) {
                best.valid     = true;
                best.gain      = g;
                best.feature   = cont_cols[fi];
                best.bin       = b;
                best.threshold = 0.5 * (bin_edges[fi][b] + bin_edges[fi][b + 1]);
                best.is_binary = false;
            }
        }
    }
    return best;
}

// ── Binary split evaluation ───────────────────────────────────────────────────

PartitionTree::SplitResult PartitionTree::evaluate_binary_splits(
    const std::vector<int>& indices,
    const Eigen::MatrixXd& X_z,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target) const
{
    const int n_node = static_cast<int>(indices.size());
    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (binary_cols.empty()) return best;

    double sum_p = 0.0, sum_sq_p = 0.0;
    for (int idx : indices) {
        double t = target[idx];
        sum_p   += t;
        sum_sq_p+= t * t;
    }

    for (int fc : binary_cols) {
        // Threshold at 0 (features are whitened, binary: ~0 or ~1)
        double sum_l = 0.0, sum_sq_l = 0.0;
        int    cnt_l = 0;
        for (int idx : indices) {
            if (X_z(idx, fc) <= 0.0) {
                sum_l    += target[idx];
                sum_sq_l += target[idx] * target[idx];
                ++cnt_l;
            }
        }
        double g = variance_gain(sum_p, sum_sq_p, n_node,
                                 sum_l, sum_sq_l, cnt_l);
        if (g > best.gain) {
            best.valid     = true;
            best.gain      = g;
            best.feature   = fc;
            best.threshold = 0.0;
            best.is_binary = true;
        }
    }
    return best;
}

// ── Build ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::build(
    const Eigen::MatrixXd& X_z,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target,
    const HVRTConfig& cfg)
{
    const int n = static_cast<int>(X_z.rows());
    d_full_ = static_cast<int>(X_z.cols());
    const int d_cont = static_cast<int>(cont_cols.size());

    // Determine limits
    int max_leaves  = cfg.n_partitions;
    int msl         = cfg.min_samples_leaf;

    if (cfg.auto_tune) {
        auto [ml, ms] = auto_tune_params(n, d_full_, /*for_reduction=*/true);
        max_leaves = ml;
        msl        = ms;
    }

    // Cache bin edges for apply()
    cont_cols_cached_   = cont_cols;
    binary_cols_cached_ = binary_cols;
    // Extract bin edges from X_binned range (need them for apply threshold lookup)
    // We store threshold in the node directly, so bin_edges only needed during build.
    // Build a local bin-edge structure from X_z / X_binned:
    // The binner was already called; we receive a pre-computed X_binned.
    // For apply, we only need the threshold stored in the node.

    // Initialize feature importances
    feature_importances_.assign(d_full_, 0.0);

    // Root node covers all samples
    nodes_.clear();
    nodes_.reserve(2 * max_leaves);
    nodes_.push_back(TreeNode{});  // root = node 0

    // Build bin_edges for each continuous feature from unique sorted values in X_z
    // (needed for evaluate_continuous_splits)
    std::vector<Eigen::VectorXd> bin_edges(d_cont);
    for (int fi = 0; fi < d_cont; ++fi) {
        int fc = cont_cols[fi];
        // Collect unique sorted edges from X_binned indices → map back through X_z
        // Since we have X_binned and X_z, edges are X_z quantiles.
        // Simple approach: gather unique X_z values for this feature, sort, build edges.
        std::vector<double> vals(n);
        for (int i = 0; i < n; ++i) vals[i] = X_z(i, fc);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        // Build edges: min, then n_bins-1 quantiles, then max
        int nb = std::min(cfg.n_bins, static_cast<int>(vals.size()));
        Eigen::VectorXd edges(nb + 1);
        edges[0] = vals.front();
        for (int b = 1; b <= nb; ++b) {
            int pos = static_cast<int>(std::round(
                static_cast<double>(b) / nb * (static_cast<int>(vals.size()) - 1)));
            pos = std::clamp(pos, 0, static_cast<int>(vals.size()) - 1);
            edges[b] = vals[pos];
        }
        bin_edges[fi] = edges;
    }
    bin_edges_ = bin_edges;
    n_bins_cached_ = cfg.n_bins;

    // BFS queue: (node_index, sample_indices)
    struct QueueEntry {
        int node_idx;
        std::vector<int> indices;
        int depth;
    };

    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::queue<QueueEntry> bfs;
    bfs.push({0, std::move(all_indices), 0});

    int leaf_count = 0;
    Eigen::VectorXi partition_ids(n);
    partition_ids.fill(-1);

    // Tracking for max gain normalization
    double total_gain = 0.0;
    std::vector<std::pair<int,double>> gain_log; // (feature, gain)

    while (!bfs.empty()) {
        auto [node_idx, indices, depth] = std::move(bfs.front());
        bfs.pop();

        int n_node = static_cast<int>(indices.size());
        TreeNode& node = nodes_[node_idx];

        bool can_split = (n_node >= 2 * msl) &&
                         (depth < cfg.max_depth) &&
                         (leaf_count + static_cast<int>(bfs.size()) + 1 < max_leaves);

        if (!can_split) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        }

        // Evaluate both streams
        SplitResult cont_split = evaluate_continuous_splits(
            indices, X_binned, cont_cols, target, bin_edges, cfg.n_bins);
        SplitResult bin_split = evaluate_binary_splits(
            indices, X_z, binary_cols, target);

        // Choose best
        SplitResult chosen;
        if (!cont_split.valid && !bin_split.valid) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        } else if (!cont_split.valid) {
            chosen = bin_split;
        } else if (!bin_split.valid) {
            chosen = cont_split;
        } else {
            chosen = (bin_split.gain > cont_split.gain) ? bin_split : cont_split;
        }

        // Check min_samples_leaf on both sides
        std::vector<int> left_idx, right_idx;
        left_idx.reserve(n_node);
        right_idx.reserve(n_node);
        for (int idx : indices) {
            double val = X_z(idx, chosen.feature);
            if (val <= chosen.threshold) left_idx.push_back(idx);
            else                         right_idx.push_back(idx);
        }

        if (static_cast<int>(left_idx.size()) < msl ||
            static_cast<int>(right_idx.size()) < msl) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        }

        // Commit split
        node.feature_idx = chosen.feature;
        node.threshold   = chosen.threshold;
        node.is_binary   = chosen.is_binary;

        feature_importances_[chosen.feature] += chosen.gain;
        total_gain += chosen.gain;

        int left_node  = static_cast<int>(nodes_.size());
        int right_node = left_node + 1;
        node.left  = left_node;
        node.right = right_node;
        nodes_.push_back(TreeNode{});
        nodes_.push_back(TreeNode{});

        bfs.push({left_node,  std::move(left_idx),  depth + 1});
        bfs.push({right_node, std::move(right_idx), depth + 1});
    }

    n_leaves_ = leaf_count;

    // Normalise feature importances
    if (total_gain > 0.0) {
        for (auto& fi : feature_importances_) fi /= total_gain;
    }

    fitted_ = true;
    return partition_ids;
}

// ── Apply ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::apply(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("PartitionTree not fitted");
    const int n = static_cast<int>(X_z.rows());
    Eigen::VectorXi ids(n);

    for (int i = 0; i < n; ++i) {
        int node_idx = 0;
        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& nd = nodes_[node_idx];
            double val = X_z(i, nd.feature_idx);
            node_idx = (val <= nd.threshold) ? nd.left : nd.right;
        }
        ids[i] = nodes_[node_idx].partition_id;
    }
    return ids;
}

} // namespace hvrt
