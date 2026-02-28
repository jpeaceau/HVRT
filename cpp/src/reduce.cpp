#include "hvrt/reduce.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>
#include "pcg_random.hpp"

namespace hvrt {

// ── Budget allocation ─────────────────────────────────────────────────────────

Eigen::VectorXi compute_budgets(
    const Eigen::VectorXi& part_ids,
    int n_target,
    int min_per_part,
    bool var_weighted,
    const Eigen::MatrixXd& X_z)
{
    int n = static_cast<int>(part_ids.size());
    int n_parts = part_ids.maxCoeff() + 1;

    // Count sizes
    std::vector<int> sizes(n_parts, 0);
    for (int i = 0; i < n; ++i) ++sizes[part_ids[i]];

    // Compute weights
    std::vector<double> weights(n_parts, 0.0);
    if (var_weighted) {
        // w[p] = mean(mean(|X_z[p]|, axis=1))
        std::vector<double> sum_w(n_parts, 0.0);
        int d = static_cast<int>(X_z.cols());
        for (int i = 0; i < n; ++i) {
            int p = part_ids[i];
            double row_mean_abs = X_z.row(i).cwiseAbs().mean();
            sum_w[p] += row_mean_abs;
        }
        for (int p = 0; p < n_parts; ++p) {
            if (sizes[p] > 0)
                weights[p] = sum_w[p] / sizes[p];
        }
    } else {
        for (int p = 0; p < n_parts; ++p)
            weights[p] = static_cast<double>(sizes[p]);
    }

    double total_w = 0.0;
    for (double w : weights) total_w += w;

    // Floor allocation
    Eigen::VectorXi budgets(n_parts);
    budgets.fill(0);

    if (total_w < 1e-12) {
        // Fallback: uniform distribution
        for (int p = 0; p < n_parts; ++p)
            budgets[p] = std::max(min_per_part,
                                  n_target / n_parts + (p < n_target % n_parts ? 1 : 0));
        return budgets;
    }

    int allocated = 0;
    std::vector<double> frac(n_parts);
    for (int p = 0; p < n_parts; ++p) {
        double exact = weights[p] / total_w * n_target;
        int floor_val = static_cast<int>(std::floor(exact));
        budgets[p] = std::max(min_per_part, floor_val);
        allocated += budgets[p];
        frac[p] = exact - floor_val;
    }

    // Greedy ±1 correction to hit exact total
    int diff = n_target - allocated;
    if (diff > 0) {
        // Need to add `diff` more — pick partitions with largest fractional remainders
        std::vector<int> order(n_parts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return frac[a] > frac[b]; });
        for (int k = 0; k < diff && k < n_parts; ++k) {
            // Only add if partition has enough samples
            int p = order[k];
            if (budgets[p] < sizes[p]) ++budgets[p];
        }
    } else if (diff < 0) {
        // Over-allocated — subtract from partitions with smallest frac (and above min)
        std::vector<int> order(n_parts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return frac[a] < frac[b]; });
        int excess = -diff;
        for (int k = 0; k < excess && k < n_parts; ++k) {
            int p = order[k];
            if (budgets[p] > min_per_part) --budgets[p];
        }
    }

    // Final clamp to partition sizes
    for (int p = 0; p < n_parts; ++p)
        budgets[p] = std::min(budgets[p], sizes[p]);

    return budgets;
}

// ── CentroidFPS ───────────────────────────────────────────────────────────────

std::vector<int> centroid_fps(const Eigen::MatrixXd& X_part, int budget) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // Seed: argmin(||X_i - centroid||²)
    Eigen::VectorXd centroid = X_part.colwise().mean();
    Eigen::VectorXd sq_dists = (X_part.rowwise() - centroid.transpose())
                                    .rowwise().squaredNorm();
    int seed_idx;
    sq_dists.minCoeff(&seed_idx);

    std::vector<int> selected;
    selected.reserve(budget);
    selected.push_back(seed_idx);

    // min_sq_dists[i] = min squared distance from i to any selected point
    std::vector<double> min_sq(n, std::numeric_limits<double>::max());
    // Update for seed
    for (int i = 0; i < n; ++i) {
        double d2 = (X_part.row(i) - X_part.row(seed_idx)).squaredNorm();
        min_sq[i] = d2;
    }

    for (int iter = 1; iter < budget; ++iter) {
        // Select argmax of min_sq_dists
        int next = static_cast<int>(
            std::max_element(min_sq.begin(), min_sq.end()) - min_sq.begin());
        selected.push_back(next);
        min_sq[next] = 0.0;

        // Update min_sq
        for (int i = 0; i < n; ++i) {
            if (min_sq[i] == 0.0) continue;
            double d2 = (X_part.row(i) - X_part.row(next)).squaredNorm();
            if (d2 < min_sq[i]) min_sq[i] = d2;
        }
    }
    return selected;
}

// ── MedoidFPS ─────────────────────────────────────────────────────────────────

static int find_medoid_exact(const Eigen::MatrixXd& X_part) {
    int n = static_cast<int>(X_part.rows());
    Eigen::VectorXd sum_dists = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double d = (X_part.row(i) - X_part.row(j)).norm();
            sum_dists[i] += d;
            sum_dists[j] += d;
        }
    int idx;
    sum_dists.minCoeff(&idx);
    return idx;
}

static int find_medoid_approx(const Eigen::MatrixXd& X_part) {
    int n = static_cast<int>(X_part.rows());
    int k = std::max(30, static_cast<int>(std::sqrt(static_cast<double>(n))));
    k = std::min(k, n);

    // k nearest to centroid
    Eigen::VectorXd centroid = X_part.colwise().mean();
    Eigen::VectorXd dists_to_centroid = (X_part.rowwise() - centroid.transpose())
                                             .rowwise().norm();

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b){ return dists_to_centroid[a] < dists_to_centroid[b]; });

    // Exact medoid within k-set
    Eigen::MatrixXd X_k(k, X_part.cols());
    for (int i = 0; i < k; ++i) X_k.row(i) = X_part.row(order[i]);

    Eigen::VectorXd sum_dists = Eigen::VectorXd::Zero(k);
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j) {
            double d = (X_k.row(i) - X_k.row(j)).norm();
            sum_dists[i] += d;
            sum_dists[j] += d;
        }
    int local_idx;
    sum_dists.minCoeff(&local_idx);
    return order[local_idx];
}

std::vector<int> medoid_fps(const Eigen::MatrixXd& X_part, int budget) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    int seed_idx = (n <= 200) ? find_medoid_exact(X_part) : find_medoid_approx(X_part);

    std::vector<int> selected;
    selected.reserve(budget);
    selected.push_back(seed_idx);

    std::vector<double> min_sq(n, std::numeric_limits<double>::max());
    for (int i = 0; i < n; ++i) {
        min_sq[i] = (X_part.row(i) - X_part.row(seed_idx)).squaredNorm();
    }

    for (int iter = 1; iter < budget; ++iter) {
        int next = static_cast<int>(
            std::max_element(min_sq.begin(), min_sq.end()) - min_sq.begin());
        selected.push_back(next);
        min_sq[next] = 0.0;
        for (int i = 0; i < n; ++i) {
            if (min_sq[i] == 0.0) continue;
            double d2 = (X_part.row(i) - X_part.row(next)).squaredNorm();
            if (d2 < min_sq[i]) min_sq[i] = d2;
        }
    }
    return selected;
}

// ── VarianceOrdered ───────────────────────────────────────────────────────────

std::vector<int> variance_ordered(const Eigen::MatrixXd& X_part, int budget, int k_nn) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    k_nn = std::min(k_nn, n - 1);

    std::vector<double> local_var(n, 0.0);
    std::vector<double> dist_buf(n);

    for (int i = 0; i < n; ++i) {
        // Compute distances from i to all others
        for (int j = 0; j < n; ++j)
            dist_buf[j] = (X_part.row(i) - X_part.row(j)).squaredNorm();

        // Partial sort to get k smallest (excluding self=0)
        std::nth_element(dist_buf.begin(), dist_buf.begin() + k_nn + 1, dist_buf.end());

        // Mean of k nearest distances (skip index 0 = self with dist 0)
        double s = 0.0;
        int cnt = 0;
        for (int j = 0; j < n && cnt < k_nn; ++j) {
            if (dist_buf[j] > 1e-15) { s += dist_buf[j]; ++cnt; }
        }
        // Variance proxy = variance of these k distances
        // (mean of sq dists is already sq, so compute var properly)
        double mean_d = (cnt > 0) ? s / cnt : 0.0;
        double var_d = 0.0;
        // Recompute actual distances to get variance
        for (int j = 0; j < n; ++j)
            dist_buf[j] = (X_part.row(i) - X_part.row(j)).squaredNorm();
        std::nth_element(dist_buf.begin(), dist_buf.begin() + k_nn + 1, dist_buf.end());
        cnt = 0;
        for (int j = 0; j < n && cnt < k_nn; ++j) {
            if (dist_buf[j] > 1e-15) {
                double d = std::sqrt(dist_buf[j]);
                var_d += (d - mean_d) * (d - mean_d);
                ++cnt;
            }
        }
        local_var[i] = (cnt > 1) ? var_d / (cnt - 1) : 0.0;
    }

    // Select top-budget by local_var descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + budget, order.end(),
                      [&](int a, int b){ return local_var[a] > local_var[b]; });
    return std::vector<int>(order.begin(), order.begin() + budget);
}

// ── Stratified ────────────────────────────────────────────────────────────────

std::vector<int> stratified_select(const Eigen::MatrixXd& X_part, int budget,
                                   uint64_t rng_seed) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    pcg64 rng(rng_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Assign random keys and take first `budget` by key
    std::vector<double> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = dist(rng);

    std::partial_sort(indices.begin(), indices.begin() + budget, indices.end(),
                      [&](int a, int b){ return keys[a] < keys[b]; });
    return std::vector<int>(indices.begin(), indices.begin() + budget);
}

// ── Top-level reduce ──────────────────────────────────────────────────────────

std::vector<int> reduce(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXi& part_ids,
    int n_target,
    ReductionMethod method,
    bool var_weighted,
    std::optional<int> n_parts_override,
    int random_state,
    int n_threads)
{
    const int n = static_cast<int>(X_z.rows());
    int n_parts = part_ids.maxCoeff() + 1;
    if (n_parts_override) n_parts = std::min(n_parts, *n_parts_override);

    // Budget per partition
    Eigen::VectorXi budgets = compute_budgets(
        part_ids, n_target, /*min_per_part=*/1, var_weighted, X_z);

    // Group sample indices by partition
    std::vector<std::vector<int>> part_indices(n_parts);
    for (int i = 0; i < n; ++i) {
        int p = part_ids[i];
        if (p < n_parts) part_indices[p].push_back(i);
    }

    // Per-partition selection (parallel via thread pool)
    std::vector<std::vector<int>> local_selections(n_parts);

    auto process_partition = [&](int p) {
        const std::vector<int>& pidx = part_indices[p];
        if (pidx.empty()) return;

        int bud = std::min(budgets[p], static_cast<int>(pidx.size()));
        if (bud <= 0) return;

        // Build sub-matrix
        Eigen::MatrixXd X_part(pidx.size(), X_z.cols());
        for (int k = 0; k < static_cast<int>(pidx.size()); ++k)
            X_part.row(k) = X_z.row(pidx[k]);

        std::vector<int> local_sel;
        switch (method) {
        case ReductionMethod::CentroidFPS:
            local_sel = centroid_fps(X_part, bud);
            break;
        case ReductionMethod::MedoidFPS:
            local_sel = medoid_fps(X_part, bud);
            break;
        case ReductionMethod::VarianceOrdered:
            local_sel = variance_ordered(X_part, bud);
            break;
        case ReductionMethod::Stratified:
            local_sel = stratified_select(X_part, bud,
                                          static_cast<uint64_t>(random_state) + p);
            break;
        }

        // Map local indices back to global
        local_selections[p].reserve(local_sel.size());
        for (int li : local_sel) local_selections[p].push_back(pidx[li]);
    };

    if (n_threads <= 1) {
        for (int p = 0; p < n_parts; ++p) process_partition(p);
    } else {
        ThreadPool pool(n_threads);
        std::vector<std::future<void>> futs;
        futs.reserve(n_parts);
        for (int p = 0; p < n_parts; ++p)
            futs.push_back(pool.submit(process_partition, p));
        for (auto& f : futs) f.get();
    }

    // Flatten
    std::vector<int> result;
    result.reserve(n_target);
    for (int p = 0; p < n_parts; ++p)
        for (int idx : local_selections[p])
            result.push_back(idx);

    return result;
}

} // namespace hvrt
