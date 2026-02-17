/*
 * ============================================================
 *         DYNAMIC PROGRAMMING — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  0/1 Knapsack
 *    2.  Unbounded Knapsack
 *    3.  Longest Increasing Subsequence (LIS) — O(N log N)
 *    4.  Longest Common Subsequence (LCS)
 *    5.  Edit Distance
 *    6.  Coin Change (min coins / count ways)
 *    7.  Matrix Chain Multiplication
 *    8.  Bitmask DP (TSP-style)
 *    9.  Digit DP
 *   10.  DP on Trees (rerooting)
 *   11.  SOS DP (Sum over Subsets)
 *   12.  Divide & Conquer DP Optimization
 *   13.  Knuth Optimization
 *   14.  Convex Hull Trick (CHT)
 *   15.  Matrix Exponentiation for Linear Recurrences
 *   16.  Profile DP (Broken Profile)
 *   17.  Subset Sum
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = 1e18;
const int MOD = 1e9 + 7;

// ═══════════════════════════════════════════════════════════
// 1. 0/1 KNAPSACK — O(N * W)
// ═══════════════════════════════════════════════════════════
// Returns maximum value achievable with capacity W
ll knapsack_01(int n, int W, const vector<int>& wt, const vector<ll>& val) {
    vector<ll> dp(W + 1, 0);
    for (int i = 0; i < n; i++)
        for (int j = W; j >= wt[i]; j--)
            dp[j] = max(dp[j], dp[j - wt[i]] + val[i]);
    return dp[W];
}

// ═══════════════════════════════════════════════════════════
// 2. UNBOUNDED KNAPSACK — O(N * W)
// ═══════════════════════════════════════════════════════════
ll knapsack_unbounded(int n, int W, const vector<int>& wt, const vector<ll>& val) {
    vector<ll> dp(W + 1, 0);
    for (int i = 0; i < n; i++)
        for (int j = wt[i]; j <= W; j++)
            dp[j] = max(dp[j], dp[j - wt[i]] + val[i]);
    return dp[W];
}

// ═══════════════════════════════════════════════════════════
// 3. LONGEST INCREASING SUBSEQUENCE — O(N log N)
// ═══════════════════════════════════════════════════════════
// Returns LIS length. Change lower_bound → upper_bound for non-decreasing.
int lis(const vector<int>& a) {
    vector<int> dp;
    for (int x : a) {
        auto it = lower_bound(dp.begin(), dp.end(), x);
        if (it == dp.end()) dp.push_back(x);
        else *it = x;
    }
    return dp.size();
}

// LIS with actual subsequence recovery
vector<int> lis_with_recovery(const vector<int>& a) {
    int n = a.size();
    vector<int> dp, pos, par(n, -1), idx;
    for (int i = 0; i < n; i++) {
        auto it = lower_bound(dp.begin(), dp.end(), a[i]);
        int p = it - dp.begin();
        if (it == dp.end()) { dp.push_back(a[i]); idx.push_back(i); }
        else { *it = a[i]; idx[p] = i; }
        par[i] = p > 0 ? idx[p - 1] : -1;
    }
    vector<int> result;
    for (int i = idx.back(); i != -1; i = par[i])
        result.push_back(a[i]);
    reverse(result.begin(), result.end());
    return result;
}

// ═══════════════════════════════════════════════════════════
// 4. LONGEST COMMON SUBSEQUENCE — O(N * M)
// ═══════════════════════════════════════════════════════════
int lcs(const string& a, const string& b) {
    int n = a.size(), m = b.size();
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            dp[i][j] = (a[i-1] == b[j-1]) ? dp[i-1][j-1] + 1 : max(dp[i-1][j], dp[i][j-1]);
    return dp[n][m];
}

// Space-optimized LCS — O(min(N, M)) space
int lcs_optimized(const string& a, const string& b) {
    int n = a.size(), m = b.size();
    if (n < m) return lcs_optimized(b, a);
    vector<int> prev(m + 1, 0), curr(m + 1, 0);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++)
            curr[j] = (a[i-1] == b[j-1]) ? prev[j-1] + 1 : max(prev[j], curr[j-1]);
        swap(prev, curr);
        fill(curr.begin(), curr.end(), 0);
    }
    return prev[m];
}

// ═══════════════════════════════════════════════════════════
// 5. EDIT DISTANCE — O(N * M)
// ═══════════════════════════════════════════════════════════
int edit_distance(const string& a, const string& b) {
    int n = a.size(), m = b.size();
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    for (int i = 0; i <= n; i++) dp[i][0] = i;
    for (int j = 0; j <= m; j++) dp[0][j] = j;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            dp[i][j] = (a[i-1] == b[j-1]) ? dp[i-1][j-1] :
                        1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
    return dp[n][m];
}

// ═══════════════════════════════════════════════════════════
// 6. COIN CHANGE
// ═══════════════════════════════════════════════════════════
// Minimum coins to make amount
int coin_change_min(const vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int c : coins)
        for (int j = c; j <= amount; j++)
            dp[j] = min(dp[j], dp[j - c] + 1);
    return dp[amount] > amount ? -1 : dp[amount];
}

// Count ways to make amount
ll coin_change_ways(const vector<int>& coins, int amount) {
    vector<ll> dp(amount + 1, 0);
    dp[0] = 1;
    for (int c : coins)
        for (int j = c; j <= amount; j++)
            dp[j] = (dp[j] + dp[j - c]) % MOD;
    return dp[amount];
}

// ═══════════════════════════════════════════════════════════
// 7. MATRIX CHAIN MULTIPLICATION — O(N³)
// ═══════════════════════════════════════════════════════════
// dims[i] = dimension of matrix i (dims[i-1] x dims[i])
ll matrix_chain(const vector<int>& dims) {
    int n = dims.size() - 1;
    vector<vector<ll>> dp(n, vector<ll>(n, 0));
    for (int len = 2; len <= n; len++)
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;
            for (int k = i; k < j; k++)
                dp[i][j] = min(dp[i][j],
                    dp[i][k] + dp[k+1][j] + (ll)dims[i]*dims[k+1]*dims[j+1]);
        }
    return dp[0][n-1];
}

// ═══════════════════════════════════════════════════════════
// 8. BITMASK DP — TSP (Travelling Salesman) — O(2^N * N²)
// ═══════════════════════════════════════════════════════════
// dist[i][j] = distance from city i to city j
ll tsp(int n, const vector<vector<ll>>& dist) {
    int full = (1 << n) - 1;
    vector<vector<ll>> dp(1 << n, vector<ll>(n, INF));
    dp[1][0] = 0; // start from city 0
    for (int mask = 1; mask <= full; mask++) {
        for (int last = 0; last < n; last++) {
            if (!(mask & (1 << last)) || dp[mask][last] == INF) continue;
            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                int nmask = mask | (1 << next);
                dp[nmask][next] = min(dp[nmask][next], dp[mask][last] + dist[last][next]);
            }
        }
    }
    ll ans = INF;
    for (int i = 0; i < n; i++)
        ans = min(ans, dp[full][i] + dist[i][0]);
    return ans;
}

// ═══════════════════════════════════════════════════════════
// 9. DIGIT DP — Count numbers in [1, N] with property P
// ═══════════════════════════════════════════════════════════
/*
 * Generic framework. Customize the state and transition.
 * Example: count numbers ≤ N whose digit sum = S
 */
ll digit_dp_example(const string& num, int target_sum) {
    int n = num.size();
    // dp[pos][sum][tight][started]
    map<tuple<int,int,bool,bool>, ll> memo;

    function<ll(int, int, bool, bool)> solve =
        [&](int pos, int sum, bool tight, bool started) -> ll {
        if (pos == n) return (started && sum == target_sum) ? 1 : 0;
        auto key = make_tuple(pos, sum, tight, started);
        if (memo.count(key)) return memo[key];

        int limit = tight ? (num[pos] - '0') : 9;
        ll res = 0;
        for (int d = 0; d <= limit; d++) {
            res += solve(pos + 1, sum + d,
                         tight && (d == limit),
                         started || (d > 0));
        }
        return memo[key] = res;
    };
    return solve(0, 0, true, false);
}

// ═══════════════════════════════════════════════════════════
// 10. DP ON TREES — Rerooting Technique
// ═══════════════════════════════════════════════════════════
/*
 * Calculate f(v) = answer when v is root, for all v.
 * Phase 1: Root at node 0, compute dp[v] bottom-up.
 * Phase 2: Reroot — compute answer for each node.
 *
 * Example: max distance from any node in tree
 */
struct RerootDP {
    int n;
    vector<vector<int>> adj;
    vector<ll> dp, ans;

    RerootDP(int n) : n(n), adj(n), dp(n, 0), ans(n, 0) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Phase 1: compute dp[v] = max depth in subtree of v
    void dfs1(int u, int p) {
        dp[u] = 0;
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs1(v, u);
            dp[u] = max(dp[u], dp[v] + 1);
        }
    }

    // Phase 2: reroot — ans[u] = answer when u is root
    void dfs2(int u, int p, ll up_val) {
        // Gather all child contributions
        vector<ll> children;
        for (int v : adj[u])
            if (v != p) children.push_back(dp[v] + 1);

        // top2 = two largest child values (for rerooting)
        ll mx1 = -1, mx2 = -1;
        for (ll c : children) {
            if (c >= mx1) { mx2 = mx1; mx1 = c; }
            else if (c > mx2) mx2 = c;
        }

        ans[u] = max(up_val, mx1 == -1 ? 0 : mx1);

        for (int v : adj[u]) {
            if (v == p) continue;
            ll child_val = dp[v] + 1;
            // new_up = best path through u not going into v's subtree
            ll new_up = 1 + max(up_val, (child_val == mx1 ? mx2 : mx1));
            if (mx1 == -1 && up_val == -1) new_up = 1; // leaf edge case
            dfs2(v, u, max(new_up, (ll)0));
        }
    }

    void solve() {
        dfs1(0, -1);
        dfs2(0, -1, 0);
    }
};

// ═══════════════════════════════════════════════════════════
// 11. SOS DP (Sum Over Subsets) — O(N * 2^N)
// ═══════════════════════════════════════════════════════════
/*
 * Given f[mask], compute for each mask:
 *   g[mask] = sum of f[submask] for all submask ⊆ mask
 */
void sos_dp(vector<ll>& f, int N) {
    // N = number of bits
    for (int i = 0; i < N; i++)
        for (int mask = 0; mask < (1 << N); mask++)
            if (mask & (1 << i))
                f[mask] += f[mask ^ (1 << i)];
    // Now f[mask] = sum over all subsets of mask
}

// Inverse SOS (Mobius inversion): subtract to get back
void sos_dp_inverse(vector<ll>& f, int N) {
    for (int i = 0; i < N; i++)
        for (int mask = 0; mask < (1 << N); mask++)
            if (mask & (1 << i))
                f[mask] -= f[mask ^ (1 << i)];
}

// ═══════════════════════════════════════════════════════════
// 12. DIVIDE & CONQUER DP OPTIMIZATION — O(N * M * log N)
// ═══════════════════════════════════════════════════════════
/*
 * Applicable when dp[i][j] = min over k<j of { dp[i-1][k] + C(k+1, j) }
 * and the optimal k is monotone (opt[j] <= opt[j+1])
 */
// cost(l, r) = cost function (must satisfy quadrangle inequality)
// dp[i][j] = min cost to split first j items into i groups
ll cost_fn(int l, int r); // Define your cost function

void dc_dp_solve(int i, int lo, int hi, int opt_lo, int opt_hi,
                  vector<vector<ll>>& dp) {
    if (lo > hi) return;
    int mid = (lo + hi) / 2;
    int opt = opt_lo;
    dp[i][mid] = INF;
    for (int k = opt_lo; k <= min(mid - 1, opt_hi); k++) {
        ll val = dp[i-1][k] + cost_fn(k + 1, mid);
        if (val < dp[i][mid]) {
            dp[i][mid] = val;
            opt = k;
        }
    }
    dc_dp_solve(i, lo, mid - 1, opt_lo, opt, dp);
    dc_dp_solve(i, mid + 1, hi, opt, opt_hi, dp);
}

// ═══════════════════════════════════════════════════════════
// 13. KNUTH OPTIMIZATION — O(N²) instead of O(N³)
// ═══════════════════════════════════════════════════════════
/*
 * For dp[i][j] = min over i<k<j of { dp[i][k] + dp[k][j] + C(i,j) }
 * where C satisfies quadrangle inequality.
 * opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
 */
void knuth_optimization(int n, vector<vector<ll>>& dp, vector<vector<int>>& opt,
                         const function<ll(int,int)>& C) {
    // dp and opt should be (n+1) x (n+1)
    for (int i = 0; i <= n; i++) dp[i][i] = 0, opt[i][i] = i;
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len <= n; i++) {
            int j = i + len;
            dp[i][j] = INF;
            for (int k = opt[i][j-1]; k <= opt[i+1][j]; k++) {
                ll val = dp[i][k] + dp[k][j] + C(i, j);
                if (val < dp[i][j]) {
                    dp[i][j] = val;
                    opt[i][j] = k;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// 14. CONVEX HULL TRICK (CHT) — for dp[i] = min(a_j * x_i + b_j)
// ═══════════════════════════════════════════════════════════
struct ConvexHullTrick {
    // For minimum queries. For maximum, negate slopes and intercepts.
    struct Line {
        ll m, b; // y = m*x + b
        ll eval(ll x) const { return m * x + b; }
    };
    deque<Line> hull;

    // Returns true if line l2 is unnecessary given l1 and l3
    bool bad(const Line& l1, const Line& l2, const Line& l3) {
        // Intersection of l1&l3 is at or before l1&l2
        return (__int128)(l3.b - l1.b) * (l1.m - l2.m) <=
               (__int128)(l2.b - l1.b) * (l1.m - l3.m);
    }

    // Add line y = m*x + b (slopes should be added in decreasing order for min)
    void add_line(ll m, ll b) {
        Line l = {m, b};
        while (hull.size() >= 2 && bad(hull[hull.size()-2], hull[hull.size()-1], l))
            hull.pop_back();
        hull.push_back(l);
    }

    // Query minimum value at x (x should be increasing)
    ll query(ll x) {
        while (hull.size() >= 2 && hull[0].eval(x) >= hull[1].eval(x))
            hull.pop_front();
        return hull.front().eval(x);
    }
};

// Li Chao Tree — more general CHT (arbitrary query order)
struct LiChaoTree {
    struct Line { ll m = 0, b = INF; ll eval(ll x) const { return m * x + b; } };
    int n;
    vector<Line> tree;

    LiChaoTree(int n) : n(n), tree(4 * n) {}

    void add_line(Line new_line, int node, int lo, int hi) {
        int mid = (lo + hi) / 2;
        bool left_better = new_line.eval(lo) < tree[node].eval(lo);
        bool mid_better  = new_line.eval(mid) < tree[node].eval(mid);
        if (mid_better) swap(tree[node], new_line);
        if (lo == hi) return;
        if (left_better != mid_better) add_line(new_line, 2*node, lo, mid);
        else add_line(new_line, 2*node+1, mid+1, hi);
    }

    void add_line(ll m, ll b) { add_line({m, b}, 1, 0, n - 1); }

    ll query(int x, int node, int lo, int hi) {
        ll res = tree[node].eval(x);
        if (lo == hi) return res;
        int mid = (lo + hi) / 2;
        if (x <= mid) return min(res, query(x, 2*node, lo, mid));
        else return min(res, query(x, 2*node+1, mid+1, hi));
    }

    ll query(int x) { return query(x, 1, 0, n - 1); }
};

// ═══════════════════════════════════════════════════════════
// 15. MATRIX EXPONENTIATION — for Linear Recurrences O(K³ log N)
// ═══════════════════════════════════════════════════════════
typedef vector<vector<ll>> Matrix;

Matrix mat_mult(const Matrix& A, const Matrix& B, ll mod = MOD) {
    int n = A.size(), m = B[0].size(), k = B.size();
    Matrix C(n, vector<ll>(m, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            for (int p = 0; p < k; p++)
                C[i][j] = (C[i][j] + A[i][p] * B[p][j]) % mod;
    return C;
}

Matrix mat_pow(Matrix M, ll exp, ll mod = MOD) {
    int n = M.size();
    Matrix result(n, vector<ll>(n, 0));
    for (int i = 0; i < n; i++) result[i][i] = 1; // identity
    while (exp > 0) {
        if (exp & 1) result = mat_mult(result, M, mod);
        M = mat_mult(M, M, mod);
        exp >>= 1;
    }
    return result;
}

/*
 * Example: Fibonacci — F(n) = F(n-1) + F(n-2)
 * Transition matrix: [[1,1],[1,0]]
 * [F(n+1), F(n)] = [[1,1],[1,0]]^n * [F(1), F(0)]
 *
 * Usage:
 *   Matrix M = {{1,1},{1,0}};
 *   Matrix res = mat_pow(M, n);
 *   // F(n) = res[0][1] (0-indexed, F(0)=0, F(1)=1)
 */

// ═══════════════════════════════════════════════════════════
// 16. PROFILE DP (Broken Profile) — Tiling problems
// ═══════════════════════════════════════════════════════════
/*
 * Count ways to tile N x M grid with 1x2 dominoes.
 * State = bitmask of current column profile.
 * O(N * M * 2^M) — use with M ≤ ~20
 */
ll tiling_dominoes(int N, int M) {
    if (M > N) swap(N, M); // keep M small
    int full = (1 << M) - 1;
    vector<vector<ll>> dp(N + 1, vector<ll>(1 << M, 0));
    dp[0][full] = 1;
    for (int col = 0; col < N; col++) {
        for (int mask = 0; mask <= full; mask++) {
            if (dp[col][mask] == 0) continue;
            // Try to fill column 'col' given profile 'mask'
            function<void(int, int, int)> fill = [&](int row, int cur_mask, int next_mask) {
                if (row == M) {
                    dp[col + 1][next_mask] = (dp[col + 1][next_mask] + dp[col][mask]) % MOD;
                    return;
                }
                if (cur_mask & (1 << row)) {
                    // Already filled, move on
                    fill(row + 1, cur_mask, next_mask);
                } else {
                    // Place horizontal domino (extends to next column)
                    fill(row + 1, cur_mask, next_mask | (1 << row));
                    // Place vertical domino (fills current + row+1 in same column)
                    if (row + 1 < M && !(cur_mask & (1 << (row + 1))))
                        fill(row + 2, cur_mask, next_mask);
                }
            };
            fill(0, mask, 0);
        }
    }
    return dp[N][full];
}

// ═══════════════════════════════════════════════════════════
// 17. SUBSET SUM — O(N * S)
// ═══════════════════════════════════════════════════════════
// Check if subset with given sum exists
bool subset_sum(const vector<int>& a, int target) {
    int n = a.size();
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int i = 0; i < n; i++)
        for (int j = target; j >= a[i]; j--)
            dp[j] = dp[j] || dp[j - a[i]];
    return dp[target];
}

// Count subsets with given sum
ll subset_sum_count(const vector<int>& a, int target) {
    int n = a.size();
    vector<ll> dp(target + 1, 0);
    dp[0] = 1;
    for (int i = 0; i < n; i++)
        for (int j = target; j >= a[i]; j--)
            dp[j] = (dp[j] + dp[j - a[i]]) % MOD;
    return dp[target];
}

// Bitset-optimized subset sum — O(N * S / 64)
bool subset_sum_bitset(const vector<int>& a, int target) {
    bitset<100001> dp;
    dp[0] = 1;
    for (int x : a) dp |= (dp << x);
    return dp[target];
}

/*
 * ══════════════════════════════════════════
 *  DP PATTERN CHEAT SHEET
 * ══════════════════════════════════════════
 *
 * LINEAR DP:
 *   - LIS, LCS, Edit Distance, Coin Change
 *   - Think: dp[i] depends on dp[j] for j < i
 *
 * INTERVAL DP:
 *   - Matrix Chain, Burst Balloons, Palindrome Partition
 *   - Think: dp[l][r] = answer for subarray [l..r]
 *   - Iterate by length, then left endpoint
 *
 * KNAPSACK:
 *   - 0/1 → reverse inner loop
 *   - Unbounded → forward inner loop
 *   - Bounded → binary lifting trick or deque optimization
 *
 * BITMASK DP:
 *   - N ≤ 20 usually, O(2^N * N)
 *   - dp[mask] = answer considering elements in mask
 *   - Common: TSP, assignment, set cover
 *
 * DIGIT DP:
 *   - Count numbers with property in range [L, R]
 *   - State: (position, accumulated_state, tight, started)
 *   - Always do f(R) - f(L-1)
 *
 * TREE DP:
 *   - dp[v] computed from children of v
 *   - Rerooting: compute for root, then propagate
 *
 * SOS DP:
 *   - g[mask] = aggregate over all subsets of mask
 *   - O(N * 2^N) — iterate over bits
 *
 * OPTIMIZATIONS:
 *   - Divide & Conquer DP: when opt[j] is monotone
 *   - Knuth: when dp[i][j] = min(dp[i][k]+dp[k][j]+C(i,j))
 *   - CHT: when dp[i] = min(a_j * x + b_j) — lines
 *   - Aliens trick: when answer is concave/convex in k
 */
