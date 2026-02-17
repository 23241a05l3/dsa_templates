/*
 * ============================================================
 *     MATH EXTRAS — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Matrix Operations (multiply, power, det, inverse, rank)
 *    2.  Probability DP patterns
 *    3.  Aliens Trick (WQS Binary Search / Lambda Optimization)
 *    4.  Lagrange Interpolation
 *    5.  FFT (alternative to NTT for non-prime mods)
 *    6.  Burnside's Lemma
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<vector<ll>> Matrix;

const ll MOD = 1e9 + 7;

ll power(ll base, ll exp, ll mod) {
    ll res = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) res = res * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return res;
}
ll modinv(ll a, ll mod) { return power(a, mod - 2, mod); }

// ═══════════════════════════════════════════════════════════
// 1. MATRIX OPERATIONS
// ═══════════════════════════════════════════════════════════

namespace MatrixOps {

    // Create identity matrix
    Matrix identity(int n) {
        Matrix I(n, vector<ll>(n, 0));
        for (int i = 0; i < n; i++) I[i][i] = 1;
        return I;
    }

    // Matrix multiplication (mod)
    Matrix multiply(const Matrix& A, const Matrix& B, ll mod = MOD) {
        int n = A.size(), m = B[0].size(), p = B.size();
        Matrix C(n, vector<ll>(m, 0));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < p; k++) {
                if (A[i][k] == 0) continue;
                for (int j = 0; j < m; j++)
                    C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
            }
        return C;
    }

    // Matrix power — O(N³ log P)
    Matrix mat_pow(Matrix A, ll p, ll mod = MOD) {
        int n = A.size();
        Matrix result = identity(n);
        while (p > 0) {
            if (p & 1) result = multiply(result, A, mod);
            A = multiply(A, A, mod);
            p >>= 1;
        }
        return result;
    }

    // Matrix addition (mod)
    Matrix add(const Matrix& A, const Matrix& B, ll mod = MOD) {
        int n = A.size(), m = A[0].size();
        Matrix C(n, vector<ll>(m));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i][j] = (A[i][j] + B[i][j]) % mod;
        return C;
    }

    // Determinant (mod p, p must be prime for modinv)
    ll determinant(Matrix A, ll mod = MOD) {
        int n = A.size();
        ll det = 1;
        for (int col = 0; col < n; col++) {
            // Find pivot
            int pivot = -1;
            for (int row = col; row < n; row++) {
                if (A[row][col] != 0) { pivot = row; break; }
            }
            if (pivot == -1) return 0;
            if (pivot != col) {
                swap(A[pivot], A[col]);
                det = (mod - det) % mod;
            }
            det = det * A[col][col] % mod;
            ll inv = modinv(A[col][col], mod);
            for (int row = col + 1; row < n; row++) {
                ll factor = A[row][col] * inv % mod;
                for (int j = col; j < n; j++)
                    A[row][j] = (A[row][j] - factor * A[col][j] % mod + mod) % mod;
            }
        }
        return det;
    }

    // Determinant over reals (no mod)
    double determinant_real(vector<vector<double>> A) {
        int n = A.size();
        double det = 1;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (abs(A[row][col]) > abs(A[pivot][col])) pivot = row;
            if (abs(A[pivot][col]) < 1e-12) return 0;
            if (pivot != col) { swap(A[pivot], A[col]); det *= -1; }
            det *= A[col][col];
            for (int row = col + 1; row < n; row++) {
                double factor = A[row][col] / A[col][col];
                for (int j = col; j < n; j++)
                    A[row][j] -= factor * A[col][j];
            }
        }
        return det;
    }

    // Matrix Inverse (mod p) — returns empty if singular
    Matrix inverse(Matrix A, ll mod = MOD) {
        int n = A.size();
        Matrix aug(n, vector<ll>(2 * n, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int pivot = -1;
            for (int row = col; row < n; row++)
                if (aug[row][col] != 0) { pivot = row; break; }
            if (pivot == -1) return {}; // singular
            swap(aug[pivot], aug[col]);
            ll inv = modinv(aug[col][col], mod);
            for (int j = 0; j < 2 * n; j++)
                aug[col][j] = aug[col][j] * inv % mod;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                ll factor = aug[row][col];
                for (int j = 0; j < 2 * n; j++)
                    aug[row][j] = (aug[row][j] - factor * aug[col][j] % mod + mod) % mod;
            }
        }
        Matrix inv_mat(n, vector<ll>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv_mat[i][j] = aug[i][n + j];
        return inv_mat;
    }

    // Matrix Rank (mod p)
    int rank(Matrix A, ll mod = MOD) {
        int n = A.size(), m = A[0].size(), r = 0;
        for (int col = 0; col < m && r < n; col++) {
            int pivot = -1;
            for (int row = r; row < n; row++)
                if (A[row][col] != 0) { pivot = row; break; }
            if (pivot == -1) continue;
            swap(A[pivot], A[r]);
            ll inv = modinv(A[r][col], mod);
            for (int row = r + 1; row < n; row++) {
                ll factor = A[row][col] * inv % mod;
                for (int j = col; j < m; j++)
                    A[row][j] = (A[row][j] - factor * A[r][j] % mod + mod) % mod;
            }
            r++;
        }
        return r;
    }

    // Rank over reals
    int rank_real(vector<vector<double>> A) {
        int n = A.size(), m = A[0].size(), r = 0;
        for (int col = 0; col < m && r < n; col++) {
            int pivot = r;
            for (int row = r + 1; row < n; row++)
                if (abs(A[row][col]) > abs(A[pivot][col])) pivot = row;
            if (abs(A[pivot][col]) < 1e-12) continue;
            swap(A[pivot], A[r]);
            for (int row = r + 1; row < n; row++) {
                double factor = A[row][col] / A[r][col];
                for (int j = col; j < m; j++)
                    A[row][j] -= factor * A[r][j];
            }
            r++;
        }
        return r;
    }

    /*
     * ─── MATRIX EXPONENTIATION USAGE ───
     *
     * Linear recurrence: f(n) = c1*f(n-1) + c2*f(n-2) + ... + ck*f(n-k)
     * Transition matrix:
     *   | c1 c2 c3 ... ck |   | f(n-1) |   | f(n)   |
     *   | 1  0  0  ... 0  | * | f(n-2) | = | f(n-1) |
     *   | 0  1  0  ... 0  |   | f(n-3) |   | f(n-2) |
     *   | ...             |   | ...    |   | ...    |
     *   | 0  0  0  ... 0  |   | f(n-k) |   | f(n-k+1)|
     *
     * Answer = (T^(n-k) * base_vector)[0]
     *
     * Fibonacci: T = {{1,1},{1,0}}, base = {F1, F0} = {1, 0}
     * nth Fibonacci = (T^n * {1, 0})[1]  (or equivalently T^(n-1) * {1, 0})[0])
     */
}

// ═══════════════════════════════════════════════════════════
// 2. PROBABILITY DP PATTERNS
// ═══════════════════════════════════════════════════════════

/*
 * Pattern 1: Expected value computation
 * E[X] = sum over i of (i * P(X = i))
 * E[X] = sum of P(X >= i) for non-negative integer X
 *
 * Pattern 2: Linearity of expectation
 * E[sum] = sum of E[each] (even if not independent!)
 *
 * Pattern 3: dp on probabilities
 * dp[state] = probability of reaching state
 * dp[state] = sum over prev_states of (dp[prev] * transition_prob)
 */

// Example: Expected number of coin flips to get N heads in a row
double expected_n_heads(int n) {
    // E[n] = 2 * E[n-1] + 2
    // E[0] = 0, E[1] = 2, E[2] = 6, E[3] = 14, ...
    // E[n] = 2^(n+1) - 2
    double e = 0;
    for (int i = 1; i <= n; i++)
        e = 2 * e + 2;
    return e;
}

// Example: Coupon Collector — expected draws to collect all N distinct coupons
// E = N * H(N) where H(N) = 1 + 1/2 + 1/3 + ... + 1/N
double coupon_collector(int n) {
    double e = 0;
    for (int i = 1; i <= n; i++)
        e += (double)n / i;
    return e;
}

// Expected value with modular arithmetic
// Use modinv for probability fractions
// E[state] values are stored as modular fractions
// If P = a/b, store as a * modinv(b, MOD) % MOD

// Example: Random walk on [0, N], step right with prob p, left with prob 1-p
// Expected steps to reach N from 0
// Uses Gaussian elimination or recurrence with modular inverse

// General probability DP template with mod arithmetic
// dp[i] = expected steps from state i to terminal
// dp[terminal] = 0
// dp[i] = 1 + sum(prob[j] * dp[j]) for transitions i → j
// Rearrange: dp[i] - sum(prob[j] * dp[j]) = 1
// This gives a system of linear equations → Gaussian elimination

// ═══════════════════════════════════════════════════════════
// 3. ALIENS TRICK (WQS Binary Search / Lambda Optimization)
// ═══════════════════════════════════════════════════════════
/*
 * PROBLEM TYPE:
 *   Minimize f(k) = cost of optimal solution using EXACTLY k items/segments
 *   But f(k) is convex in k (or concave — then negate)
 *
 * KEY IDEA:
 *   Instead of constraining "exactly k", add a penalty lambda per item.
 *   g(lambda) = min over all k of (f(k) + lambda * k)
 *   Binary search on lambda to find the right k.
 *
 * WHEN f(k) is convex₳
 *   g(lambda) is computable without the constraint on k
 *   Binary search lambda to find value where optimal k = target k
 *
 * IMPLEMENTATION:
 *   1. Binary search on lambda in [lo, hi]
 *   2. For each lambda, solve the unconstrained problem:
 *      "minimize total_cost + lambda * count"
 *      This gives optimal count c and cost
 *   3. If c > target_k → increase lambda (penalty too low)
 *      If c < target_k → decrease lambda
 *      If c == target_k → found answer = cost - lambda * k
 */

// Template for Aliens trick
// solve_unconstrained(lambda) returns {min_cost_with_penalty, count_used}
// We want exactly target_k items
ll aliens_trick(int target_k, function<pair<ll, int>(ll)> solve_unconstrained) {
    ll lo = -1e18, hi = 1e18; // adjust range based on problem
    ll best_cost = 0;

    while (lo <= hi) {
        ll mid = lo + (hi - lo) / 2;
        auto [cost, count] = solve_unconstrained(mid);
        if (count <= target_k) {
            best_cost = cost - mid * target_k;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return best_cost;
}

// ═══════════════════════════════════════════════════════════
// 4. LAGRANGE INTERPOLATION — O(N²) or O(N) for consecutive
// ═══════════════════════════════════════════════════════════

// Given n points (x[i], y[i]), find f(target) where f is the unique
// polynomial of degree <= n-1 passing through all points.
// O(N²) general, works mod prime
ll lagrange_interpolation(const vector<ll>& x, const vector<ll>& y, ll target, ll mod = MOD) {
    int n = x.size();
    ll result = 0;
    for (int i = 0; i < n; i++) {
        ll num = y[i] % mod, den = 1;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            num = num % mod * ((target - x[j]) % mod + mod) % mod;
            den = den * ((x[i] - x[j]) % mod + mod) % mod;
        }
        result = (result + num % mod * modinv(den, mod)) % mod;
    }
    return (result + mod) % mod;
}

// O(N) Lagrange for CONSECUTIVE x-values: x = {0, 1, 2, ..., n-1}
// Given y[0..n-1], find f(target)
ll lagrange_consecutive(const vector<ll>& y, ll target, ll mod = MOD) {
    int n = y.size();
    if (target >= 0 && target < n) return y[target]; // direct lookup

    // prefix[i] = (target - 0)(target - 1)...(target - (i-1))
    // suffix[i] = (target - (i+1))(target - (i+2))...(target - (n-1))
    vector<ll> prefix(n + 1, 1), suffix(n + 1, 1);
    for (int i = 0; i < n; i++)
        prefix[i + 1] = prefix[i] % mod * ((target - i) % mod + mod) % mod;
    for (int i = n - 1; i >= 0; i--)
        suffix[i] = suffix[i + 1] % mod * ((target - i) % mod + mod) % mod;

    // Precompute factorial inverses
    vector<ll> fact(n), inv_fact(n);
    fact[0] = 1;
    for (int i = 1; i < n; i++) fact[i] = fact[i-1] * i % mod;
    inv_fact[n-1] = modinv(fact[n-1], mod);
    for (int i = n - 2; i >= 0; i--) inv_fact[i] = inv_fact[i+1] * (i + 1) % mod;

    ll result = 0;
    for (int i = 0; i < n; i++) {
        ll num = prefix[i] % mod * suffix[i + 1] % mod;
        ll den = inv_fact[i] % mod * inv_fact[n - 1 - i] % mod;
        if ((n - 1 - i) % 2 == 1) den = (mod - den) % mod;
        result = (result + y[i] % mod * num % mod * den) % mod;
    }
    return (result + mod) % mod;
}

// ═══════════════════════════════════════════════════════════
// 5. FFT (Fast Fourier Transform) — for non-prime mod
// ═══════════════════════════════════════════════════════════
using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd>& a, bool invert) {
    int n = a.size();
    if (n == 1) return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) for (auto& x : a) x /= n;
}

// Multiply two polynomials using FFT — result coefficients are real
vector<ll> multiply_fft(const vector<int>& a, const vector<int>& b) {
    vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < (int)(a.size() + b.size())) n <<= 1;
    fa.resize(n); fb.resize(n);
    fft(fa, false); fft(fb, false);
    for (int i = 0; i < n; i++) fa[i] *= fb[i];
    fft(fa, true);
    vector<ll> res(n);
    for (int i = 0; i < n; i++) res[i] = llround(fa[i].real());
    return res;
}

// Multiply with mod — split into high/low parts to avoid precision issues
vector<ll> multiply_fft_mod(const vector<int>& a, const vector<int>& b, ll mod) {
    int n = 1;
    while (n < (int)(a.size() + b.size())) n <<= 1;
    const int SPLIT = 1 << 15; // sqrt(mod)
    // Split a[i] = a_hi * SPLIT + a_lo, same for b
    vector<cd> a_lo(n), a_hi(n), b_lo(n), b_hi(n);
    for (int i = 0; i < (int)a.size(); i++) { a_lo[i] = a[i] % SPLIT; a_hi[i] = a[i] / SPLIT; }
    for (int i = 0; i < (int)b.size(); i++) { b_lo[i] = b[i] % SPLIT; b_hi[i] = b[i] / SPLIT; }
    fft(a_lo, false); fft(a_hi, false); fft(b_lo, false); fft(b_hi, false);

    vector<cd> c1(n), c2(n), c3(n);
    for (int i = 0; i < n; i++) {
        c1[i] = a_lo[i] * b_lo[i]; // lo*lo
        c2[i] = a_lo[i] * b_hi[i] + a_hi[i] * b_lo[i]; // cross
        c3[i] = a_hi[i] * b_hi[i]; // hi*hi
    }
    fft(c1, true); fft(c2, true); fft(c3, true);

    vector<ll> res(n);
    for (int i = 0; i < n; i++) {
        ll v1 = llround(c1[i].real()) % mod;
        ll v2 = llround(c2[i].real()) % mod;
        ll v3 = llround(c3[i].real()) % mod;
        res[i] = (v1 + v2 % mod * SPLIT + v3 % mod * SPLIT % mod * SPLIT) % mod;
    }
    return res;
}

// ═══════════════════════════════════════════════════════════
// 6. BURNSIDE'S LEMMA
// ═══════════════════════════════════════════════════════════
/*
 * Count distinct objects under group of symmetries.
 * |distinct| = (1/|G|) * sum over g in G of |fix(g)|
 * where fix(g) = number of objects unchanged by symmetry g
 *
 * Example: necklaces with n beads, k colors
 *   Rotational symmetry group has n elements (rotate by 0,1,...,n-1)
 *   fix(rotation by r) = k^(gcd(n, r))
 *   |distinct necklaces| = (1/n) * sum_{r=0}^{n-1} k^gcd(n,r)
 */
ll count_necklaces(int n, int k) {
    // Number of distinct necklaces with n beads, k colors (rotations only)
    ll total = 0;
    for (int r = 0; r < n; r++)
        total = (total + power(k, __gcd(n, r), MOD)) % MOD;
    return total % MOD * modinv(n, MOD) % MOD;
}

ll count_bracelets(int n, int k) {
    // Necklaces considering both rotation AND reflection
    // Dihedral group of order 2n
    ll total = 0;
    // Rotations
    for (int r = 0; r < n; r++)
        total = (total + power(k, __gcd(n, r), MOD)) % MOD;
    // Reflections
    if (n % 2 == 1) {
        // All reflections: axis through one vertex and midpoint of opposite edge
        total = (total + (ll)n % MOD * power(k, (n + 1) / 2, MOD)) % MOD;
    } else {
        // n/2 axes through pairs of vertices: k^(n/2+1)
        // n/2 axes through midpoints of edges: k^(n/2)
        total = (total + (ll)(n / 2) % MOD * power(k, n / 2 + 1, MOD) % MOD +
                 (ll)(n / 2) % MOD * power(k, n / 2, MOD)) % MOD;
    }
    return total % MOD * modinv(2 * n, MOD) % MOD;
}
