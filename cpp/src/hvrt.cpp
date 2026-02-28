#include "hvrt/hvrt.h"
#include "hvrt/target.h"
#include "hvrt/reduce.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <set>

namespace hvrt {

// ── Constructor ───────────────────────────────────────────────────────────────

HVRT::HVRT(HVRTConfig cfg) : cfg_(std::move(cfg)) {}

// ── Feature type detection ────────────────────────────────────────────────────

std::vector<bool> HVRT::detect_feature_types(
    const Eigen::MatrixXd& X,
    const std::optional<std::vector<std::string>>& feature_types) const
{
    const int d = static_cast<int>(X.cols());
    const int n = static_cast<int>(X.rows());
    std::vector<bool> cat_mask(d, false);

    if (feature_types) {
        if (static_cast<int>(feature_types->size()) != d)
            throw std::invalid_argument("feature_types length must match X columns");
        for (int j = 0; j < d; ++j)
            cat_mask[j] = ((*feature_types)[j] == "categorical");
    } else {
        // Auto-detect: if a column has <= 10 unique values AND all are integers → categorical
        for (int j = 0; j < d; ++j) {
            std::set<double> uniq;
            bool all_int = true;
            for (int i = 0; i < n; ++i) {
                double v = X(i, j);
                uniq.insert(v);
                if (std::abs(v - std::round(v)) > 1e-9) all_int = false;
            }
            cat_mask[j] = (all_int && static_cast<int>(uniq.size()) <= 10);
        }
    }
    return cat_mask;
}

// ── Parse enums ───────────────────────────────────────────────────────────────

ReductionMethod HVRT::parse_reduction_method(const std::string& s) {
    if (s == "centroid_fps" || s == "CentroidFPS") return ReductionMethod::CentroidFPS;
    if (s == "medoid_fps"   || s == "MedoidFPS")   return ReductionMethod::MedoidFPS;
    if (s == "variance"     || s == "VarianceOrdered") return ReductionMethod::VarianceOrdered;
    if (s == "stratified"   || s == "Stratified")  return ReductionMethod::Stratified;
    throw std::invalid_argument("Unknown reduction method: " + s);
}

GenerationStrategy HVRT::parse_generation_strategy(const std::string& s) {
    if (s == "auto"          || s == "Auto")            return GenerationStrategy::Auto;
    if (s == "epanechnikov"  || s == "Epanechnikov")    return GenerationStrategy::Epanechnikov;
    if (s == "multivariate_kde" || s == "MultivariateKDE") return GenerationStrategy::MultivariateKDE;
    if (s == "bootstrap"     || s == "BootstrapNoise")  return GenerationStrategy::BootstrapNoise;
    if (s == "copula"        || s == "UnivariateCopula") return GenerationStrategy::UnivariateCopula;
    throw std::invalid_argument("Unknown generation strategy: " + s);
}

// ── Fit ───────────────────────────────────────────────────────────────────────

HVRT& HVRT::fit(
    const Eigen::MatrixXd& X,
    std::optional<Eigen::VectorXd> y,
    std::optional<std::vector<std::string>> feature_types)
{
    const int n = static_cast<int>(X.rows());
    const int d = static_cast<int>(X.cols());

    if (n < 2) throw std::invalid_argument("Need at least 2 samples");
    if (d < 1) throw std::invalid_argument("Need at least 1 feature");

    X_orig_ = X;

    // 1. Feature type detection
    std::vector<bool> cat_mask = detect_feature_types(X, feature_types);

    cont_cols_.clear();
    cat_cols_.clear();
    for (int j = 0; j < d; ++j) {
        if (cat_mask[j]) cat_cols_.push_back(j);
        else             cont_cols_.push_back(j);
    }

    // 2. Whitener fit + transform
    whitener_.fit(X, cat_mask);
    X_z_ = whitener_.transform(X);

    // 3. Binner fit on continuous columns only
    Eigen::MatrixXd X_cont(n, static_cast<int>(cont_cols_.size()));
    for (int fi = 0; fi < static_cast<int>(cont_cols_.size()); ++fi)
        X_cont.col(fi) = X_z_.col(cont_cols_[fi]);

    std::vector<bool> binary_mask = binner_.fit(X_cont, cfg_.n_bins);

    // Identify binary vs non-binary continuous cols
    std::vector<int> binary_cols, hist_cont_cols;
    for (int fi = 0; fi < static_cast<int>(cont_cols_.size()); ++fi) {
        if (binary_mask[fi]) binary_cols.push_back(cont_cols_[fi]);
        else                 hist_cont_cols.push_back(cont_cols_[fi]);
    }

    // Rebuild X_cont without binary cols for binning
    int d_cont_hist = static_cast<int>(hist_cont_cols.size());
    Eigen::MatrixXd X_cont_hist(n, d_cont_hist);
    for (int fi = 0; fi < d_cont_hist; ++fi)
        X_cont_hist.col(fi) = X_z_.col(hist_cont_cols[fi]);

    // Re-fit binner on non-binary cols only (binary handled separately)
    Binner hist_binner;
    using BinMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    BinMat X_binned;
    if (d_cont_hist > 0) {
        hist_binner.fit(X_cont_hist, cfg_.n_bins);
        X_binned = hist_binner.transform(X_cont_hist);
    } else {
        X_binned.resize(n, 0);
    }

    // 4. Target computation
    int fast_threshold = 50;  // use pairwise for d <= threshold
    Eigen::VectorXd target_vec;
    if (d <= fast_threshold) {
        target_vec = compute_pairwise_target(X_z_);
    } else {
        target_vec = compute_sum_target(X_z_);
    }

    // 5. Y-weight blending
    if (y && cfg_.y_weight > 0.0f) {
        target_vec = blend_target(target_vec, *y, static_cast<double>(cfg_.y_weight));
    }

    // 6. Build partition tree
    // Store bin_edges in tree builder context via hist_binner
    // The tree build receives X_binned indexed by hist_cont_cols position
    partition_ids_ = tree_.build(
        X_z_, X_binned, hist_cont_cols, binary_cols, target_vec, cfg_);

    // 7. Prepare expander
    expander_.prepare(
        X_z_, partition_ids_,
        cont_cols_, cat_cols_,
        GenerationStrategy::Auto,
        cfg_.bandwidth,
        cfg_.n_threads);

    fitted_ = true;
    return *this;
}

// ── unique_partitions ─────────────────────────────────────────────────────────

std::vector<int> HVRT::unique_partitions() const {
    std::set<int> seen;
    for (int i = 0; i < static_cast<int>(partition_ids_.size()); ++i)
        seen.insert(partition_ids_[i]);
    return std::vector<int>(seen.begin(), seen.end());
}

// ── reduce_indices ────────────────────────────────────────────────────────────

std::vector<int> HVRT::reduce_indices(
    std::optional<int>    n_opt,
    std::optional<double> ratio_opt,
    const std::string&    method,
    bool var_weighted,
    std::optional<int>    n_parts) const
{
    if (!fitted_) throw std::runtime_error("HVRT not fitted");

    int n_train = static_cast<int>(X_z_.rows());
    int n_target;
    if (n_opt) {
        n_target = *n_opt;
    } else if (ratio_opt) {
        n_target = static_cast<int>(std::round(*ratio_opt * n_train));
    } else {
        n_target = n_train / 2;
    }
    n_target = std::clamp(n_target, 1, n_train);

    ReductionMethod rm = parse_reduction_method(method);
    return hvrt::reduce(X_z_, partition_ids_, n_target, rm,
                        var_weighted, n_parts,
                        cfg_.random_state, cfg_.n_threads);
}

Eigen::MatrixXd HVRT::reduce(
    std::optional<int>    n,
    std::optional<double> ratio,
    const std::string&    method,
    bool var_weighted,
    std::optional<int>    n_parts) const
{
    std::vector<int> idx = reduce_indices(n, ratio, method, var_weighted, n_parts);
    Eigen::MatrixXd result(static_cast<int>(idx.size()), X_orig_.cols());
    for (int k = 0; k < static_cast<int>(idx.size()); ++k)
        result.row(k) = X_orig_.row(idx[k]);
    return result;
}

// ── expand ────────────────────────────────────────────────────────────────────

Eigen::MatrixXd HVRT::expand(
    int n,
    bool var_weighted,
    std::optional<float> bandwidth,
    const std::string& strategy,
    bool adaptive_bandwidth,
    std::optional<int> n_parts) const
{
    if (!fitted_) throw std::runtime_error("HVRT not fitted");

    int n_partitions = partition_ids_.maxCoeff() + 1;
    if (n_parts) n_partitions = std::min(n_partitions, *n_parts);

    // Compute budgets
    Eigen::VectorXi budgets = compute_budgets(
        partition_ids_, n, /*min_per_part=*/0, var_weighted, X_z_);

    // If strategy or bandwidth differs from prepare() defaults, re-prepare
    // (For now, expander was prepared with Auto strategy during fit)
    // A production impl would cache and re-prepare on param change.

    Eigen::MatrixXd X_z_syn = expander_.generate(budgets, cfg_.random_state);

    // Inverse transform back to original space
    return whitener_.inverse_transform(X_z_syn);
}

// ── augment ───────────────────────────────────────────────────────────────────

Eigen::MatrixXd HVRT::augment(
    int n,
    bool var_weighted,
    std::optional<int> n_parts) const
{
    Eigen::MatrixXd syn = expand(n, var_weighted, std::nullopt, "auto", false, n_parts);
    Eigen::MatrixXd result(X_orig_.rows() + syn.rows(), X_orig_.cols());
    result.topRows(X_orig_.rows())   = X_orig_;
    result.bottomRows(syn.rows())    = syn;
    return result;
}

// ── get_partitions ────────────────────────────────────────────────────────────

std::vector<PartitionInfo> HVRT::get_partitions() const {
    if (!fitted_) throw std::runtime_error("HVRT not fitted");

    int n_parts = partition_ids_.maxCoeff() + 1;
    std::vector<PartitionInfo> infos(n_parts);

    std::vector<int>    sizes(n_parts, 0);
    std::vector<double> sum_abs_z(n_parts, 0.0);
    std::vector<double> sum_sq(n_parts, 0.0);
    std::vector<double> sum_val(n_parts, 0.0);

    const int n = static_cast<int>(X_z_.rows());
    const int d = static_cast<int>(X_z_.cols());

    for (int i = 0; i < n; ++i) {
        int p = partition_ids_[i];
        ++sizes[p];
        double row_mean_abs = X_z_.row(i).cwiseAbs().mean();
        sum_abs_z[p] += row_mean_abs;
        // Variance: track sum and sum_sq of row L2 norms
        double norm = X_z_.row(i).squaredNorm();
        sum_sq[p]  += norm;
        sum_val[p] += std::sqrt(norm);
    }

    for (int p = 0; p < n_parts; ++p) {
        infos[p].id   = p;
        infos[p].size = sizes[p];
        infos[p].mean_abs_z = (sizes[p] > 0) ? sum_abs_z[p] / sizes[p] : 0.0;
        // variance of row norms within partition
        double mean_norm = (sizes[p] > 0) ? sum_val[p] / sizes[p] : 0.0;
        double mean_sq   = (sizes[p] > 0) ? sum_sq[p]  / sizes[p] : 0.0;
        infos[p].variance = mean_sq - mean_norm * mean_norm;
    }
    return infos;
}

// ── compute_novelty ───────────────────────────────────────────────────────────

Eigen::VectorXd HVRT::compute_novelty(const Eigen::MatrixXd& X_new) const {
    if (!fitted_) throw std::runtime_error("HVRT not fitted");

    Eigen::MatrixXd X_new_z = whitener_.transform(X_new);
    const int n_new   = static_cast<int>(X_new_z.rows());
    const int n_train = static_cast<int>(X_z_.rows());

    Eigen::VectorXd min_dists(n_new);
    const int chunk = 256;  // process in chunks to bound memory

    for (int i = 0; i < n_new; ++i) {
        double min_d2 = std::numeric_limits<double>::max();
        for (int j = 0; j < n_train; j += chunk) {
            int end = std::min(j + chunk, n_train);
            Eigen::VectorXd d2 = (X_z_.middleRows(j, end - j).rowwise() -
                                  X_new_z.row(i))
                                     .rowwise().squaredNorm();
            double local_min = d2.minCoeff();
            if (local_min < min_d2) min_d2 = local_min;
        }
        min_dists[i] = std::sqrt(min_d2);
    }
    return min_dists;
}

// ── apply ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi HVRT::apply(const Eigen::MatrixXd& X_new) const {
    if (!fitted_) throw std::runtime_error("HVRT not fitted");
    return tree_.apply(whitener_.transform(X_new));
}

// ── to_z ──────────────────────────────────────────────────────────────────────

Eigen::MatrixXd HVRT::to_z(const Eigen::MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("HVRT not fitted");
    return whitener_.transform(X);
}

// ── recommend_params ──────────────────────────────────────────────────────────

ParamRecommendation HVRT::recommend_params(const Eigen::MatrixXd& X) {
    int n = static_cast<int>(X.rows());
    int d = static_cast<int>(X.cols());
    auto [max_leaf, msl] = PartitionTree::auto_tune_params(n, d, /*for_reduction=*/true);
    return {max_leaf, msl};
}

} // namespace hvrt
