#include "hvrt/binner.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace hvrt {

std::vector<bool> Binner::fit(const Eigen::MatrixXd& X_z_cont, int n_bins) {
    if (n_bins < 2) throw std::invalid_argument("n_bins must be >= 2");
    n_bins_    = n_bins;
    const int n = static_cast<int>(X_z_cont.rows());
    const int d = static_cast<int>(X_z_cont.cols());

    edges_.resize(d);
    binary_mask_.assign(d, false);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int f = 0; f < d; ++f) {
        // Column data pointer (Eigen::MatrixXd is column-major, so col is contiguous)
        const double* col_ptr = X_z_cont.col(f).data();

        // Fast binary detection: early-exit once a 3rd distinct value is found.
        // Avoids O(n log n) sort + O(n) unique for continuous columns, which
        // almost always resolve in the first few rows.
        double first = col_ptr[0];
        double second = first;
        int n_unique_fast = 1;
        for (int i = 1; i < n; ++i) {
            double v = col_ptr[i];
            if (v != first && v != second) {
                if (n_unique_fast == 1) { second = v; n_unique_fast = 2; }
                else { n_unique_fast = 3; break; }
            }
        }

        if (n_unique_fast <= 2) {
            binary_mask_[f] = true;
            double lo = first, hi = first;
            if (n_unique_fast == 2) {
                lo = std::min(first, second);
                hi = std::max(first, second);
            }
            // Use 3 edges [lo, mid, hi] so the tree builder's prefix scan
            // can evaluate the binary split (nb = 2 → 1 iteration).
            // With only 2 edges, nb = 1, and the scan does 0 iterations,
            // silently skipping the feature entirely.
            edges_[f].resize(3);
            edges_[f][0] = lo;
            edges_[f][1] = (lo + hi) * 0.5;
            edges_[f][2] = hi;
            continue;
        }

        // Continuous column: single bulk copy for both unique-count and quantiles.
        // memcpy from contiguous column data — compiler can vectorize this.
        std::vector<double> work(n);
        std::memcpy(work.data(), col_ptr, n * sizeof(double));

        // Sort to count exact uniques (needed for b = min(n_bins, n_unique))
        std::sort(work.begin(), work.end());
        int n_unique = static_cast<int>(
            std::unique(work.begin(), work.end()) - work.begin());

        int b = std::min(n_bins, n_unique);
        edges_[f].resize(b + 1);

        // Refresh work from source for nth_element (sort destroyed order)
        std::memcpy(work.data(), col_ptr, n * sizeof(double));

        // Compute b+1 quantile positions via O(n) nth_element per quantile
        for (int q = 0; q <= b; ++q) {
            int pos = static_cast<int>(std::round(
                static_cast<double>(q) / b * (n - 1)));
            pos = std::clamp(pos, 0, n - 1);
            std::nth_element(work.begin(), work.begin() + pos, work.end());
            edges_[f][q] = work[pos];
        }

        // Ensure strict monotonicity (collapse duplicates)
        std::vector<double> uniq_edges;
        uniq_edges.push_back(edges_[f][0]);
        for (int q = 1; q <= b; ++q) {
            if (edges_[f][q] > uniq_edges.back() + 1e-15) {
                uniq_edges.push_back(edges_[f][q]);
            }
        }
        if (uniq_edges.size() < 2) {
            uniq_edges = {work[0], work[n - 1]};
        }
        edges_[f] = Eigen::Map<Eigen::VectorXd>(
            uniq_edges.data(), static_cast<int>(uniq_edges.size()));
    }

    fitted_ = true;
    return binary_mask_;
}

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Binner::transform(const Eigen::MatrixXd& X_z_cont) const {
    if (!fitted_) throw std::runtime_error("Binner not fitted");
    const int n = static_cast<int>(X_z_cont.rows());
    const int d = static_cast<int>(X_z_cont.cols());

    using BinMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    BinMat out(n, d);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int f = 0; f < d; ++f) {
        const Eigen::VectorXd& e = edges_[f];
        int nb = static_cast<int>(e.size()) - 1; // number of bins

        if (binary_mask_[f]) {
            // binary: 0 if value <= first edge, else 1
            double thresh = (e[0] + e[1]) * 0.5;
            for (int i = 0; i < n; ++i) {
                out(i, f) = (X_z_cont(i, f) > thresh) ? 1u : 0u;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double v = X_z_cont(i, f);
                // upper_bound gives first edge > v → bin = that index - 1
                auto it = std::upper_bound(e.data(), e.data() + e.size(), v);
                int bin = static_cast<int>(it - e.data()) - 1;
                bin = std::clamp(bin, 0, nb - 1);
                out(i, f) = static_cast<uint8_t>(bin);
            }
        }
    }
    return out;
}

double Binner::get_threshold(int feat, int bin) const {
    if (!fitted_) throw std::runtime_error("Binner not fitted");
    const Eigen::VectorXd& e = edges_[feat];
    int nb = static_cast<int>(e.size()) - 1;
    if (bin < 0 || bin >= nb) throw std::out_of_range("bin out of range");
    return 0.5 * (e[bin] + e[bin + 1]);
}

} // namespace hvrt
