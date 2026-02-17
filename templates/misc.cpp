/*
 * ============================================================
 *        MISCELLANEOUS & ADVANCED TECHNIQUES — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Coordinate Compression
 *    2.  Mo's Algorithm (Offline Range Queries)
 *    3.  Mo's Algorithm on Trees
 *    4.  Sqrt Decomposition
 *    5.  Custom Hash (for unordered_map)
 *    6.  Bitwise Tricks & Utilities
 *    7.  Random Number Generation
 *    8.  Ternary Search
 *    9.  Binary Search on Answer
 *   10.  Two Pointers / Sliding Window
 *   11.  Meet in the Middle
 *   12.  Small to Large Merging (DSU on Tree)
 *   13.  Geometry Basics
 *   14.  Convex Hull
 *   15.  Common STL Tricks & Snippets
 *   16.  Fast I/O
 *   17.  Stress Testing Template
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
typedef pair<int,int> pii;
const int MOD = 1e9 + 7;
const ll INF = 1e18;

// ═══════════════════════════════════════════════════════════
// 1. COORDINATE COMPRESSION — O(N log N)
// ═══════════════════════════════════════════════════════════
struct CoordCompress {
    vector<int> vals;

    void add(int x) { vals.push_back(x); }

    void build() {
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
    }

    int compress(int x) { // 0-indexed
        return lower_bound(vals.begin(), vals.end(), x) - vals.begin();
    }

    int decompress(int idx) { return vals[idx]; }

    int size() { return vals.size(); }
};

// Quick inline version:
// vector<int> sorted_vals = vals;
// sort(sorted_vals.begin(), sorted_vals.end());
// sorted_vals.erase(unique(sorted_vals.begin(), sorted_vals.end()), sorted_vals.end());
// for (int& x : vals) x = lower_bound(sorted_vals.begin(), sorted_vals.end(), x) - sorted_vals.begin();

// ═══════════════════════════════════════════════════════════
// 2. MO'S ALGORITHM — O((N + Q) * sqrt(N))
// ═══════════════════════════════════════════════════════════
struct MoQuery {
    int l, r, idx;
};

// Template — customize add/remove/get_answer
struct Mo {
    int block_size;
    vector<int> a;
    int cur_answer; // or whatever type

    Mo(const vector<int>& a) : a(a), cur_answer(0) {
        block_size = max(1, (int)sqrt(a.size()));
    }

    void add(int idx) {
        // Add a[idx] to current window
        // Example: cur_answer += a[idx];
    }

    void remove(int idx) {
        // Remove a[idx] from current window
        // Example: cur_answer -= a[idx];
    }

    vector<int> solve(vector<MoQuery>& queries) {
        // Sort queries
        sort(queries.begin(), queries.end(), [&](const MoQuery& a, const MoQuery& b) {
            int ba = a.l / block_size, bb = b.l / block_size;
            if (ba != bb) return ba < bb;
            return (ba & 1) ? (a.r > b.r) : (a.r < b.r); // Hilbert-curve-like ordering
        });

        int cur_l = 0, cur_r = -1;
        vector<int> answers(queries.size());

        for (auto& q : queries) {
            while (cur_r < q.r) add(++cur_r);
            while (cur_l > q.l) add(--cur_l);
            while (cur_r > q.r) remove(cur_r--);
            while (cur_l < q.l) remove(cur_l++);
            answers[q.idx] = cur_answer;
        }
        return answers;
    }
};

// ═══════════════════════════════════════════════════════════
// 3. MO'S ON TREES — using Euler Tour
// ═══════════════════════════════════════════════════════════
/*
 * Flatten tree using Euler Tour (entry/exit times).
 * For path queries: use LCA + two ranges from Euler tour.
 * For subtree queries: single range [tin[v], tout[v]).
 */

// ═══════════════════════════════════════════════════════════
// 4. SQRT DECOMPOSITION — O(sqrt(N)) per query
// ═══════════════════════════════════════════════════════════
struct SqrtDecomp {
    int n, block;
    vector<int> a;
    vector<ll> block_sum;

    SqrtDecomp(const vector<int>& a) : a(a), n(a.size()) {
        block = max(1, (int)sqrt(n));
        block_sum.assign((n + block - 1) / block, 0);
        for (int i = 0; i < n; i++)
            block_sum[i / block] += a[i];
    }

    void update(int idx, int val) {
        block_sum[idx / block] += val - a[idx];
        a[idx] = val;
    }

    ll query(int l, int r) {
        ll sum = 0;
        int lb = l / block, rb = r / block;
        if (lb == rb) {
            for (int i = l; i <= r; i++) sum += a[i];
        } else {
            for (int i = l; i < (lb + 1) * block; i++) sum += a[i];
            for (int b = lb + 1; b < rb; b++) sum += block_sum[b];
            for (int i = rb * block; i <= r; i++) sum += a[i];
        }
        return sum;
    }
};

// ═══════════════════════════════════════════════════════════
// 5. CUSTOM HASH — anti-hack for unordered_map
// ═══════════════════════════════════════════════════════════
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
    // For pairs:
    size_t operator()(pair<int,int> x) const {
        return splitmix64(x.first * 31 + x.second + chrono::steady_clock::now().time_since_epoch().count());
    }
};
// Usage: unordered_map<int, int, custom_hash> mp;
// Usage: unordered_set<int, custom_hash> st;

// Even faster: gp_hash_table from pb_ds
// #include <ext/pb_ds/assoc_container.hpp>
// __gnu_pbds::gp_hash_table<int, int, custom_hash> mp;

// ═══════════════════════════════════════════════════════════
// 6. BITWISE TRICKS & UTILITIES
// ═══════════════════════════════════════════════════════════
namespace BitTricks {
    // Count set bits
    int popcount(int x) { return __builtin_popcount(x); }
    int popcount(ll x)  { return __builtin_popcountll(x); }

    // Least significant bit
    int lsb(int x) { return x & (-x); }

    // Most significant bit position (0-indexed)
    int msb_pos(int x) { return 31 - __builtin_clz(x); }
    int msb_pos(ll x)  { return 63 - __builtin_clzll(x); }

    // Check if power of 2
    bool is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

    // Next higher number with same popcount (Gosper's hack)
    int next_set_of(int x) {
        int c = x & (-x), r = x + c;
        return (((r ^ x) >> 2) / c) | r;
    }

    // Iterate over all submasks of mask (excluding 0)
    // for (int sub = mask; sub > 0; sub = (sub - 1) & mask) { ... }

    // Iterate over all supersets of mask in [0, 2^n)
    // for (int sup = mask; sup < (1<<n); sup = (sup + 1) | mask) { ... }

    // Gray code
    int gray(int n) { return n ^ (n >> 1); }
    int inv_gray(int g) { int n = 0; for (; g; g >>= 1) n ^= g; return n; }

    // Bit reversal for NTT
    int bit_reverse(int x, int bits) {
        int result = 0;
        for (int i = 0; i < bits; i++) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
}

// ═══════════════════════════════════════════════════════════
// 7. RANDOM NUMBER GENERATION
// ═══════════════════════════════════════════════════════════
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
mt19937_64 rng64(chrono::steady_clock::now().time_since_epoch().count());

int rand_int(int l, int r) { return uniform_int_distribution<int>(l, r)(rng); }
ll rand_ll(ll l, ll r) { return uniform_int_distribution<ll>(l, r)(rng64); }

// Shuffle
template<typename T>
void shuffle_vec(vector<T>& v) { shuffle(v.begin(), v.end(), rng); }

// ═══════════════════════════════════════════════════════════
// 8. TERNARY SEARCH — O(log N) — for unimodal functions
// ═══════════════════════════════════════════════════════════
// Find x in [lo, hi] that MINIMIZES f(x) (assumes f is unimodal)
// For integers:
ll ternary_search_int(ll lo, ll hi, function<ll(ll)> f) {
    while (hi - lo > 2) {
        ll m1 = lo + (hi - lo) / 3;
        ll m2 = hi - (hi - lo) / 3;
        if (f(m1) < f(m2)) hi = m2;
        else lo = m1;
    }
    ll best = lo;
    for (ll i = lo; i <= hi; i++)
        if (f(i) < f(best)) best = i;
    return best;
}

// For floating point:
ld ternary_search_real(ld lo, ld hi, function<ld(ld)> f, int iterations = 200) {
    for (int i = 0; i < iterations; i++) {
        ld m1 = lo + (hi - lo) / 3;
        ld m2 = hi - (hi - lo) / 3;
        if (f(m1) < f(m2)) hi = m2;
        else lo = m1;
    }
    return (lo + hi) / 2;
}

// ═══════════════════════════════════════════════════════════
// 9. BINARY SEARCH ON ANSWER — common pattern
// ═══════════════════════════════════════════════════════════
/*
 * Template for "find minimum x such that check(x) is true"
 * where check is monotone (false,false,...,false,true,true,...,true)
 */
ll binary_search_answer(ll lo, ll hi, function<bool(ll)> check) {
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (check(mid)) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Floating point binary search
ld binary_search_real(ld lo, ld hi, function<bool(ld)> check, int iterations = 200) {
    for (int i = 0; i < iterations; i++) {
        ld mid = (lo + hi) / 2;
        if (check(mid)) hi = mid;
        else lo = mid;
    }
    return (lo + hi) / 2;
}

// ═══════════════════════════════════════════════════════════
// 10. TWO POINTERS / SLIDING WINDOW — O(N)
// ═══════════════════════════════════════════════════════════
/*
 * Pattern 1: Longest subarray with condition
 *   int l = 0, ans = 0;
 *   for (int r = 0; r < n; r++) {
 *       // Add a[r] to window
 *       while (window is invalid) {
 *           // Remove a[l] from window
 *           l++;
 *       }
 *       ans = max(ans, r - l + 1);
 *   }
 *
 * Pattern 2: Count subarrays with condition (e.g., sum <= K)
 *   ll cnt = 0; int l = 0;
 *   for (int r = 0; r < n; r++) {
 *       // Add a[r]
 *       while (condition violated) { remove a[l]; l++; }
 *       cnt += (r - l + 1); // all subarrays ending at r
 *   }
 *
 * Pattern 3: Two arrays (merge-style)
 *   int i = 0, j = 0;
 *   while (i < n && j < m) { ... }
 */

// ═══════════════════════════════════════════════════════════
// 11. MEET IN THE MIDDLE — O(2^(N/2) * log)
// ═══════════════════════════════════════════════════════════
// Example: count subsets with sum <= S in array of size up to 40
ll meet_in_middle(const vector<int>& a, ll S) {
    int n = a.size();
    int half = n / 2;

    // Generate all subset sums for first half
    vector<ll> left_sums;
    for (int mask = 0; mask < (1 << half); mask++) {
        ll sum = 0;
        for (int i = 0; i < half; i++)
            if (mask & (1 << i)) sum += a[i];
        left_sums.push_back(sum);
    }
    sort(left_sums.begin(), left_sums.end());

    // For each subset of second half, binary search in first half
    ll count = 0;
    int other = n - half;
    for (int mask = 0; mask < (1 << other); mask++) {
        ll sum = 0;
        for (int i = 0; i < other; i++)
            if (mask & (1 << i)) sum += a[half + i];
        if (sum > S) continue;
        count += upper_bound(left_sums.begin(), left_sums.end(), S - sum) - left_sums.begin();
    }
    return count;
}

// ═══════════════════════════════════════════════════════════
// 12. SMALL TO LARGE MERGING (DSU on Tree / Euler Tour + merge)
// ═══════════════════════════════════════════════════════════
// Count distinct values in each subtree — O(N log N)
struct DSUonTree {
    int n;
    vector<vector<int>> adj;
    vector<int> val, sz;
    vector<set<int>> sets;
    vector<int> ans; // answer for each node

    DSUonTree(int n) : n(n), adj(n), val(n), sz(n, 1), sets(n), ans(n) {}

    void dfs(int u, int p) {
        sets[u].insert(val[u]);
        int heavy = -1;
        for (int v : adj[u]) {
            if (v == p) continue;
            dfs(v, u);
            if (heavy == -1 || sets[v].size() > sets[heavy].size())
                heavy = v;
        }
        if (heavy != -1) swap(sets[u], sets[heavy]); // steal largest
        for (int v : adj[u]) {
            if (v == p || v == heavy) continue;
            for (int x : sets[v]) sets[u].insert(x);
        }
        ans[u] = sets[u].size();
    }
};

// ═══════════════════════════════════════════════════════════
// 13. GEOMETRY BASICS
// ═══════════════════════════════════════════════════════════
typedef complex<ld> Point;
#define X real()
#define Y imag()

ld cross(Point a, Point b) { return a.X * b.Y - a.Y * b.X; }
ld dot(Point a, Point b) { return a.X * b.X + a.Y * b.Y; }
ld dist(Point a, Point b) { return abs(a - b); }

// Orientation: >0 left, <0 right, ==0 collinear
ld orientation(Point a, Point b, Point c) { return cross(b - a, c - a); }

// Point on segment
bool on_segment(Point p, Point a, Point b) {
    return abs(cross(b - a, p - a)) < 1e-9 &&
           dot(p - a, p - b) <= 1e-9;
}

// Line intersection: returns true + intersection point
bool line_intersect(Point a, Point b, Point c, Point d, Point& res) {
    ld d1 = cross(d - c, a - c);
    ld d2 = cross(d - c, b - c);
    if (abs(d1 - d2) < 1e-9) return false; // parallel
    res = a + (b - a) * d1 / (d1 - d2);
    return true;
}

// ═══════════════════════════════════════════════════════════
// 14. CONVEX HULL — O(N log N)
// ═══════════════════════════════════════════════════════════
vector<Point> convex_hull(vector<Point> pts) {
    int n = pts.size();
    if (n < 2) return pts;
    sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        return a.X < b.X || (a.X == b.X && a.Y < b.Y);
    });
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    n = pts.size();
    if (n < 2) return pts;

    vector<Point> hull;
    // Lower hull
    for (auto& p : pts) {
        while (hull.size() >= 2 && cross(hull.back() - hull[hull.size()-2], p - hull[hull.size()-2]) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }
    // Upper hull
    int lower_sz = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while ((int)hull.size() > lower_sz && cross(hull.back() - hull[hull.size()-2], pts[i] - hull[hull.size()-2]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    hull.pop_back();
    return hull;
}

// Area of polygon (convex or not)
ld polygon_area(const vector<Point>& pts) {
    ld area = 0;
    int n = pts.size();
    for (int i = 0; i < n; i++)
        area += cross(pts[i], pts[(i+1) % n]);
    return abs(area) / 2;
}

// ═══════════════════════════════════════════════════════════
// 15. COMMON STL TRICKS & SNIPPETS
// ═══════════════════════════════════════════════════════════
namespace STLTricks {
    // Unique + sort
    template<typename T>
    void uniquify(vector<T>& v) {
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }

    // Prefix sums
    template<typename T>
    vector<T> prefix_sum(const vector<T>& a) {
        int n = a.size();
        vector<T> ps(n + 1, 0);
        for (int i = 0; i < n; i++) ps[i+1] = ps[i] + a[i];
        return ps;
        // Sum of [l, r] = ps[r+1] - ps[l]
    }

    // 2D prefix sums
    template<typename T>
    vector<vector<T>> prefix_sum_2d(const vector<vector<T>>& grid) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<T>> ps(n + 1, vector<T>(m + 1, 0));
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                ps[i][j] = grid[i-1][j-1] + ps[i-1][j] + ps[i][j-1] - ps[i-1][j-1];
        return ps;
        // Sum of rectangle (r1,c1) to (r2,c2): ps[r2+1][c2+1] - ps[r1][c2+1] - ps[r2+1][c1] + ps[r1][c1]
    }

    // Difference array — apply range add efficiently
    template<typename T>
    struct DiffArray {
        vector<T> diff;
        DiffArray(int n) : diff(n + 1, 0) {}
        void add(int l, int r, T val) { diff[l] += val; diff[r+1] -= val; }
        vector<T> build() {
            int n = diff.size() - 1;
            vector<T> result(n);
            result[0] = diff[0];
            for (int i = 1; i < n; i++) result[i] = result[i-1] + diff[i];
            return result;
        }
    };

    // Next permutation → generates all permutations
    // do { ... } while (next_permutation(v.begin(), v.end()));

    // Erase by value from multiset (single occurrence)
    // ms.erase(ms.find(val));

    // map/set lower_bound/upper_bound: O(log N)
    // For sorted vectors: use std::lower_bound (O(log N) with random access)
}

// ═══════════════════════════════════════════════════════════
// 16. FAST I/O
// ═══════════════════════════════════════════════════════════
namespace FastIO {
    // Use at the start of main:
    // ios_base::sync_with_stdio(false);
    // cin.tie(NULL);

    // For even faster I/O (manual):
    inline int readInt() {
        int x = 0, f = 1;
        char c = getchar_unlocked();
        while (c < '0' || c > '9') { if (c == '-') f = -1; c = getchar_unlocked(); }
        while (c >= '0' && c <= '9') { x = x * 10 + c - '0'; c = getchar_unlocked(); }
        return x * f;
    }

    inline ll readLL() {
        ll x = 0; int f = 1;
        char c = getchar_unlocked();
        while (c < '0' || c > '9') { if (c == '-') f = -1; c = getchar_unlocked(); }
        while (c >= '0' && c <= '9') { x = x * 10 + c - '0'; c = getchar_unlocked(); }
        return x * f;
    }

    // Bulk output buffer
    char output_buf[1 << 22];
    int output_ptr = 0;
    void flush() { fwrite(output_buf, 1, output_ptr, stdout); output_ptr = 0; }
    void writeChar(char c) { output_buf[output_ptr++] = c; }
    void writeInt(int x) {
        if (x < 0) { writeChar('-'); x = -x; }
        char buf[12]; int len = 0;
        do { buf[len++] = '0' + x % 10; x /= 10; } while (x);
        while (len--) writeChar(buf[len]);
    }
}

// ═══════════════════════════════════════════════════════════
// 17. STRESS TESTING TEMPLATE
// ═══════════════════════════════════════════════════════════
/*
 * Usage: Save brute force as brute(), optimized as solve().
 * Run this in main() to find failing test cases.
 *
 * void brute() { ... }  // slow but correct
 * void solve() { ... }  // fast solution to test
 *
 * void stress_test() {
 *     for (int test = 1; ; test++) {
 *         // Generate random input
 *         int n = rand_int(1, 10);
 *         vector<int> a(n);
 *         for (int& x : a) x = rand_int(1, 100);
 *
 *         // Run both solutions
 *         // Compare outputs
 *         // If different, print test case and break
 *
 *         if (test % 10000 == 0) cerr << "Passed " << test << " tests\n";
 *     }
 * }
 *
 * Typical stress test runner:
 *
 *   // In bash:
 *   // while true; do
 *   //   python3 gen.py > input.txt
 *   //   ./brute < input.txt > brute_out.txt
 *   //   ./sol < input.txt > sol_out.txt
 *   //   diff brute_out.txt sol_out.txt || break
 *   // done
 */

/*
 * ══════════════════════════════════════════════════════
 *  COMPETITIVE PROGRAMMING TIPS & PATTERNS
 * ══════════════════════════════════════════════════════
 *
 * READING THE PROBLEM:
 *   1. Read constraints FIRST — they hint at expected complexity
 *   2. N ≤ 20       → Bitmask DP, brute force O(2^N * N)
 *   3. N ≤ 40       → Meet in the Middle O(2^(N/2))
 *   4. N ≤ 500      → O(N³) — Floyd-Warshall, MCM
 *   5. N ≤ 5000     → O(N²) — simple DP
 *   6. N ≤ 10^5     → O(N log N) — sort, binary search, seg tree
 *   7. N ≤ 10^6     → O(N) or O(N log N) — linear algos, sieve
 *   8. N ≤ 10^9     → O(sqrt(N)) or O(log N) — math, binary search
 *   9. N ≤ 10^18    → O(log N) — matrix exponentiation, binary search
 *
 * COMMON PATTERNS:
 *   - "Minimum/Maximum something" → Binary search on answer + greedy check
 *   - "Count subarrays with X"    → Two pointers / prefix sums
 *   - "Shortest path"             → BFS/Dijkstra/Bellman-Ford
 *   - "Connected components"      → DSU / BFS / DFS
 *   - "Range queries"             → Segment Tree / BIT / Sparse Table
 *   - "Substring/pattern"         → KMP / Z / Hashing / SA
 *   - "Game theory"               → Sprague-Grundy / Nim
 *   - "XOR problems"              → Binary Trie / Linear Basis
 *   - "Grid problems"             → BFS + 4/8 directions
 *   - "Interval scheduling"       → Sort by end time + greedy
 *   - "Tree + queries"            → LCA / HLD / Centroid Decomp
 *
 * DEBUGGING CHECKLIST:
 *   □ Integer overflow? Use ll.
 *   □ Array out of bounds?
 *   □ Uninitialized variables?
 *   □ Off-by-one errors?
 *   □ Edge cases: n=0, n=1, all same, sorted, reverse sorted?
 *   □ Modular arithmetic: negative values?
 *   □ Graph: 1-indexed vs 0-indexed?
 *   □ Multi-test: reset all globals?
 *   □ Special case when answer is 0?
 *
 * TIME COMPLEXITY LIMITS (~10^8 operations/sec):
 *   1 sec → ~10^8     |  2 sec → ~2*10^8
 *   3 sec → ~3*10^8   |  5 sec → ~5*10^8
 */
