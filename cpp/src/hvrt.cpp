#include "hvrt/hvrt.h"
#include "hvrt/target.h"
#include "hvrt/reduce.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>
#include <stdexcept>
#include <chrono>
#include <cstdio>

namespace hvrt {

// ── Constructor ───────────────────────────────────────────────────────────────

HVRT::HVRT(HVRTConfig cfg) : cfg_(std::move(cfg)) {}

// ── Parse enums ───────────────────────────────────────────────────────────────

ReductionMethod HVRT::parse_reduction_method(const std::string& s) {
    if (s == "centroid_fps"        || s == "CentroidFPS")      return ReductionMethod::CentroidFPS;
    if (s == "medoid_fps"          || s == "MedoidFPS")        return ReductionMethod::MedoidFPS;
    if (s == "variance"            || s == "VarianceOrdered"
                                   || s == "variance_ordered") return ReductionMethod::VarianceOrdered;
    if (s == "stratified"          || s == "Stratified")       return ReductionMethod::Stratified;
    if (s == "orthant_stratified"  || s == "OrthantStratified") return ReductionMethod::OrthantStratified;
    throw std::invalid_argument("Unknown reduction method: " + s);
}

GenerationStrategy HVRT::parse_generation_strategy(const std::string& s) {
    if (s == "auto"             || s == "Auto")            return GenerationStrategy::Auto;
    if (s == "epanechnikov"     || s == "Epanechnikov")    return GenerationStrategy::Epanechnikov;
    if (s == "multivariate_kde" || s == "MultivariateKDE") return GenerationStrategy::MultivariateKDE;
    if (s == "bootstrap"        || s == "BootstrapNoise")  return GenerationStrategy::BootstrapNoise;
    if (s == "copula"           || s == "UnivariateCopula") return GenerationStrategy::UnivariateCopula;
    if (s == "simplex_mixup"    || s == "SimplexMixup")    return GenerationStrategy::SimplexMixup;
    if (s == "laplace"          || s == "Laplace")         return GenerationStrategy::Laplace;
    throw std::invalid_argument("Unknown generation strategy: " + s);
}

// ── Fit ───────────────────────────────────────────────────────────────────────

HVRT& HVRT::fit(
    const Eigen::MatrixXd& X,
    std::optional<Eigen::VectorXd> y,
    std::optional<std::vector<std::string>> /* feature_types — encoding is user-managed */)
{
    const int n = static_cast<int>(X.rows());
    const int d = static_cast<int>(X.cols());

    if (n < 2) throw std::invalid_argument("Need at least 2 samples");
    if (d < 1) throw std::invalid_argument("Need at least 1 feature");

    X_orig_ = X;

    // 1. Whitener: all columns treated as continuous (no categorical path).
    //    Categorical encoding is the caller's responsibility.
    whitener_.fit(X, std::vector<bool>(d, false));
    X_z_ = whitener_.transform(X);

    // 2. Detect binary columns (≤2 unique values post-whitening) via a quick
    //    Binner pass, then build X_binned for the non-binary subset.
    using BinMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    std::vector<int> binary_cols, hist_cont_cols;
    {
        Binner detect_binner;
        std::vector<bool> bm = detect_binner.fit(X_z_, cfg_.n_bins);
        for (int j = 0; j < d; ++j) {
            if (bm[j]) binary_cols.push_back(j);
            else        hist_cont_cols.push_back(j);
        }
    }

    BinMat X_binned;
    const int d_hist = static_cast<int>(hist_cont_cols.size());
    if (d_hist > 0) {
        Eigen::MatrixXd X_hist(n, d_hist);
        for (int fi = 0; fi < d_hist; ++fi)
            X_hist.col(fi) = X_z_.col(hist_cont_cols[fi]);
        binner_.fit(X_hist, cfg_.n_bins);
        X_binned = binner_.transform(X_hist);
    } else {
        X_binned.resize(n, 0);
    }

    // 3. Target computation (dispatch on partitioner_type)
    //    y participates as the (d+1)th covariate — equal contribution to all
    //    features in the pairwise/sum/hart/pyramid target.  Each feature
    //    (including y) participates in exactly d of (d+1)*d/2 pairs → weight
    //    2/(d+1), identical for every feature.  No separate blend_target step.
    //    At refit, y = residuals; as the model converges, residuals become
    //    noise → y-terms self-attenuate without explicit scheduling.
    Eigen::VectorXd target_vec;
    const bool include_y = y && y->size() == n && cfg_.y_weight > 0.0f;
    Eigen::VectorXd y_z;
    if (include_y) y_z = zscore(*y);

    // Pairwise cooperation target for all HVRT partitioning regardless of d.
    // The O(n·d²/2) cost is vectorised via Eigen column broadcasts and
    // remains sub-100 ms even at d=90, n=50k.  The cached geom_unnorm_cache_
    // makes refits O(n·d) by only recomputing d y-terms.
    const bool use_pairwise = (cfg_.partitioner_type == PartitionerType::HVRT);

    if (use_pairwise) {
        // Compute X-only unnormalized pairwise sum: Σ_{i<j} zscore(z_i * z_j)
        // Cached for efficient refit — only d y-terms need recomputing.
        geom_unnorm_cache_ = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < d - 1; ++i) {
            Eigen::MatrixXd prods = X_z_.rightCols(d - i - 1).array().colwise()
                                    * X_z_.col(i).array();
            for (int k = 0; k < prods.cols(); ++k)
                geom_unnorm_cache_ += zscore(prods.col(k));
        }

        // Full target: X-pairs + y-pairs
        Eigen::VectorXd target_unnorm = geom_unnorm_cache_;
        if (include_y) {
            for (int j = 0; j < d; ++j)
                target_unnorm += zscore(
                    (X_z_.col(j).array() * y_z.array()).matrix());
        }
        target_vec = zscore(target_unnorm);
    } else {
        // Non-pairwise partitioner (HART/FastHART/Pyramid): compute on augmented [X_z | y_z]
        geom_unnorm_cache_.resize(0);  // not used for non-pairwise path
        Eigen::MatrixXd X_z_aug;
        if (include_y) {
            X_z_aug.resize(n, d + 1);
            X_z_aug.leftCols(d) = X_z_;
            X_z_aug.col(d) = y_z;
        }
        const Eigen::MatrixXd& X_t = include_y ? X_z_aug : X_z_;

        switch (cfg_.partitioner_type) {
            case PartitionerType::HART:
                target_vec = compute_hart_target(X_t);
                break;
            case PartitionerType::FastHART:
                target_vec = compute_sum_target(X_t);
                break;
            case PartitionerType::PyramidHART:
                target_vec = compute_pyramid_target(X_t);
                break;
            default:
                target_vec = compute_sum_target(X_t);
                break;
        }
    }

    // ── Populate refit cache ───────────────────────────────────────────────────
    X_binned_cache_    = X_binned;
    hist_cont_cols_    = hist_cont_cols;
    binary_cols_       = binary_cols;

    // Cache geometry-only target (X without y) for accessor / selective_target
    if (use_pairwise) {
        geom_target_cache_ = zscore(geom_unnorm_cache_);
    } else {
        switch (cfg_.partitioner_type) {
            case PartitionerType::HART:
                geom_target_cache_ = compute_hart_target(X_z_); break;
            case PartitionerType::FastHART:
                geom_target_cache_ = compute_sum_target(X_z_); break;
            case PartitionerType::PyramidHART:
                geom_target_cache_ = compute_pyramid_target(X_z_); break;
            default:
                geom_target_cache_ = compute_sum_target(X_z_); break;
        }
    }

    // 5. Build partition tree
    partition_ids_ = tree_.build(
        X_z_, X_binned, hist_cont_cols, binary_cols, target_vec, cfg_);

    // 6. Prepare expander (all columns are continuous)
    //    Skipped when skip_expander is set (fast_refit: no expand needed).
    if (!cfg_.skip_expander) {
        std::vector<int> all_cols(d);
        std::iota(all_cols.begin(), all_cols.end(), 0);
        expander_.prepare(
            X_z_, partition_ids_,
            all_cols, /*cat_cols=*/{},
            cfg_.gen_strategy,
            cfg_.bandwidth,
            cfg_.n_threads);
    }

    fitted_ = true;
    return *this;
}

// ── Refit ─────────────────────────────────────────────────────────────────────
// Fast path: reuses cached X_z_, X_binned_cache_, and geom_target_cache_.
// Skips whitening, binning, and geometry target computation.
// Only re-runs tree_.build() (with cached bin_edges on 2nd+ call) + expander_.prepare().
// Expander is also skipped when the new partition assignments are identical to the
// previous ones (stable tree → same KDE parameters → no need to redo KDE fitting).

HVRT& HVRT::refit(std::optional<Eigen::VectorXd> y)
{
    if (!fitted_) throw std::runtime_error("HVRT::refit() called before fit()");

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Recompute target with y as (d+1)th covariate.
    const int d = static_cast<int>(X_z_.cols());
    const bool include_y = y && y->size() == X_z_.rows()
                           && cfg_.y_weight > 0.0f;
    Eigen::VectorXd target_vec;

    if (geom_unnorm_cache_.size() > 0) {
        Eigen::VectorXd target_unnorm = geom_unnorm_cache_;
        if (include_y) {
            Eigen::VectorXd y_z = zscore(*y);
            for (int j = 0; j < d; ++j)
                target_unnorm += zscore(
                    (X_z_.col(j).array() * y_z.array()).matrix());
        }
        target_vec = zscore(target_unnorm);
    } else {
        Eigen::MatrixXd X_z_aug;
        if (include_y) {
            const int n = static_cast<int>(X_z_.rows());
            X_z_aug.resize(n, d + 1);
            X_z_aug.leftCols(d) = X_z_;
            X_z_aug.col(d) = zscore(*y);
        }
        const Eigen::MatrixXd& X_t = include_y ? X_z_aug : X_z_;

        switch (cfg_.partitioner_type) {
            case PartitionerType::HART:
                target_vec = compute_hart_target(X_t); break;
            case PartitionerType::FastHART:
                target_vec = compute_sum_target(X_t); break;
            case PartitionerType::PyramidHART:
                target_vec = compute_pyramid_target(X_t); break;
            default:
                target_vec = compute_sum_target(X_t); break;
        }
    }

    auto t1 = Clock::now();

    // Re-run tree build — bin_edges_ will be reused (bin_edges_valid_ = true from fit())
    Eigen::VectorXi old_part_ids = partition_ids_;
    partition_ids_ = tree_.build(
        X_z_, X_binned_cache_, hist_cont_cols_, binary_cols_, target_vec, cfg_);

    auto t2 = Clock::now();

    // Re-prepare expander only when partition assignments changed.
    bool parts_changed = (old_part_ids.size() != partition_ids_.size())
                         || (old_part_ids != partition_ids_);
    last_refit_stable_ = !parts_changed;
    if (parts_changed && !cfg_.skip_expander) {
        const int d = static_cast<int>(X_z_.cols());
        std::vector<int> all_cols(d);
        std::iota(all_cols.begin(), all_cols.end(), 0);
        expander_.prepare(
            X_z_, partition_ids_,
            all_cols, /*cat_cols=*/{},
            cfg_.gen_strategy,
            cfg_.bandwidth,
            cfg_.n_threads);
    }

    auto t3 = Clock::now();

    // Accumulate sub-component timings
    refit_target_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    refit_tree_ms_   += std::chrono::duration<double, std::milli>(t2 - t1).count();
    refit_expand_ms_ += std::chrono::duration<double, std::milli>(t3 - t2).count();

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
    std::optional<float> /*bandwidth*/,
    const std::string& /*strategy*/,
    bool /*adaptive_bandwidth*/,
    std::optional<int> /*n_parts*/) const
{
    if (!fitted_) throw std::runtime_error("HVRT not fitted");

    // Compute budgets — do NOT clamp to partition sizes here (oversampling is fine).
    Eigen::VectorXi budgets = compute_budgets(
        partition_ids_, n, /*min_per_part=*/0, var_weighted, X_z_,
        /*clamp_to_sizes=*/false);

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
