/*
 * ============================================================
 *     GAME THEORY — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Sprague-Grundy Theorem & Nim Values
 *    2.  Nim Game (Classic)
 *    3.  Staircase Nim
 *    4.  Wythoff's Game
 *    5.  Green Hackenbush
 *    6.  XOR Basis (Linear Basis)
 *    7.  Josephus Problem — O(N) and O(K log N)
 *    8.  Impartial Games Framework
 *    9.  Common Game Patterns
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ═══════════════════════════════════════════════════════════
// 1. SPRAGUE-GRUNDY THEOREM
// ═══════════════════════════════════════════════════════════
/*
 * THEORY:
 *   Every impartial game position has a Grundy number (nim-value).
 *   Grundy(terminal state) = 0
 *   Grundy(position) = mex({Grundy(next) : next is reachable from position})
 *
 *   mex (minimum excludant) = smallest non-negative integer NOT in set
 *
 *   For combined games: Grundy(G1 + G2) = Grundy(G1) XOR Grundy(G2)
 *
 *   Position is losing (P-position) iff Grundy = 0
 *   Position is winning (N-position) iff Grundy > 0
 */

int mex(const vector<int>& s) {
    set<int> st(s.begin(), s.end());
    for (int i = 0; ; i++)
        if (st.find(i) == st.end()) return i;
}

// Generic Grundy number computation for a game
// moves(state) returns vector of states reachable from state
// WARNING: only works for DAG (no cycles in game graph)
map<int, int> grundy_cache;

int grundy(int state, function<vector<int>(int)> moves) {
    if (grundy_cache.count(state)) return grundy_cache[state];
    vector<int> next_states = moves(state);
    vector<int> next_grundy;
    for (int ns : next_states)
        next_grundy.push_back(grundy(ns, moves));
    return grundy_cache[state] = mex(next_grundy);
}

// Precompute Grundy numbers for states 0..max_val
// Good for 1D games where state is a single integer
vector<int> compute_grundy_table(int max_val, function<vector<int>(int)> moves) {
    vector<int> g(max_val + 1, 0);
    for (int s = 0; s <= max_val; s++) {
        set<int> reachable;
        for (int ns : moves(s))
            if (ns >= 0 && ns <= max_val)
                reachable.insert(g[ns]);
        int m = 0;
        while (reachable.count(m)) m++;
        g[s] = m;
    }
    return g;
}

// ═══════════════════════════════════════════════════════════
// 2. NIM GAME (Classic)
// ═══════════════════════════════════════════════════════════
/*
 * N piles of stones: a[0], a[1], ..., a[n-1]
 * Two players alternate. On each turn, remove >= 1 stone from one pile.
 * Player who takes the last stone WINS. (Normal play convention)
 *
 * First player wins iff a[0] XOR a[1] XOR ... XOR a[n-1] != 0
 *
 * Misère Nim (last stone loses):
 *   First player wins iff:
 *     XOR != 0 AND at least one pile > 1
 *     OR XOR == 0 AND all piles <= 1 (odd number of piles with 1)
 *   Simplified: if all piles are 0 or 1, win iff odd count of 1s
 *               else, same as normal except flip the answer when XOR = 0
 */
bool nim_normal(const vector<int>& piles) {
    int xor_sum = 0;
    for (int p : piles) xor_sum ^= p;
    return xor_sum != 0; // true = first player wins
}

bool nim_misere(const vector<int>& piles) {
    int xor_sum = 0;
    bool all_one_or_zero = true;
    for (int p : piles) {
        xor_sum ^= p;
        if (p > 1) all_one_or_zero = false;
    }
    if (all_one_or_zero) {
        // Win iff odd number of piles with 1 stone
        int cnt = 0;
        for (int p : piles) cnt += (p == 1);
        return cnt % 2 == 1;
    }
    return xor_sum != 0;
}

// ═══════════════════════════════════════════════════════════
// Game with restricted moves: can take 1..k stones from one pile
// Grundy(pile of size n) = n % (k + 1)
int nim_restricted(const vector<int>& piles, int k) {
    int xor_sum = 0;
    for (int p : piles) xor_sum ^= (p % (k + 1));
    return xor_sum != 0;
}

// Nim with a given set of allowed moves
// Grundy numbers computed via SG theorem
vector<int> nim_with_moves(int max_pile, const vector<int>& allowed_moves) {
    vector<int> g(max_pile + 1, 0);
    for (int n = 0; n <= max_pile; n++) {
        set<int> reachable;
        for (int m : allowed_moves)
            if (n >= m) reachable.insert(g[n - m]);
        int mex_val = 0;
        while (reachable.count(mex_val)) mex_val++;
        g[n] = mex_val;
    }
    return g;
}

// ═══════════════════════════════════════════════════════════
// 3. STAIRCASE NIM
// ═══════════════════════════════════════════════════════════
/*
 * N stairs with stones. On each turn, move >= 1 stone from stair i to stair i-1.
 * Stair 0 is ground (removed stones).
 * First player wins iff XOR of piles at ODD indices != 0.
 */
bool staircase_nim(const vector<int>& stairs) {
    int xor_sum = 0;
    for (int i = 1; i < (int)stairs.size(); i += 2)
        xor_sum ^= stairs[i];
    return xor_sum != 0;
}

// ═══════════════════════════════════════════════════════════
// 4. WYTHOFF'S GAME
// ═══════════════════════════════════════════════════════════
/*
 * Two piles. On each turn, remove any number from ONE pile,
 * or remove EQUAL number from BOTH piles.
 *
 * P-positions (cold/losing): (floor(k*phi), floor(k*phi²)) for k=0,1,2,...
 *   where phi = (1 + sqrt(5)) / 2 ≈ 1.618
 */
bool wythoff_wins(int a, int b) {
    if (a > b) swap(a, b);
    double phi = (1 + sqrt(5.0)) / 2;
    int k = b - a;
    int expected_a = (int)(k * phi);
    return !(a == expected_a); // true if first player wins
}

// ═══════════════════════════════════════════════════════════
// 5. GREEN HACKENBUSH (on trees)
// ═══════════════════════════════════════════════════════════
/*
 * Game on a graph rooted at ground. Each edge is a "stalk".
 * On each turn, remove an edge. All edges disconnected from ground are also removed.
 *
 * For TREES: Grundy value = XOR of distances to ground for each leaf path
 *   Actually: Grundy(tree rooted at v) = XOR of (1 + Grundy(child)) for each child
 */
int hackenbush_tree(int u, int parent, const vector<vector<int>>& adj) {
    int g = 0;
    for (int v : adj[u]) {
        if (v == parent) continue;
        g ^= (1 + hackenbush_tree(v, u, adj));
    }
    return g;
}

// ═══════════════════════════════════════════════════════════
// 6. XOR BASIS (LINEAR BASIS) — O(N * BITS)
// ═══════════════════════════════════════════════════════════
/*
 * Maintains a basis for a set of numbers under XOR.
 * Used for:
 *   - Maximum XOR of any subset
 *   - Check if a value can be represented as XOR of subset
 *   - Count distinct XOR values representable
 *   - Kth smallest XOR value
 */
struct XorBasis {
    static const int BITS = 60;
    ll basis[60] = {};
    int sz = 0; // number of basis vectors

    // Insert value into basis. Returns true if value was independent (added to basis).
    bool insert(ll val) {
        for (int i = BITS - 1; i >= 0; i--) {
            if (!((val >> i) & 1)) continue;
            if (!basis[i]) {
                basis[i] = val;
                sz++;
                return true;
            }
            val ^= basis[i];
        }
        return false; // linearly dependent
    }

    // Check if val is representable by the basis
    bool can_represent(ll val) {
        for (int i = BITS - 1; i >= 0; i--) {
            if (!((val >> i) & 1)) continue;
            if (!basis[i]) return false;
            val ^= basis[i];
        }
        return true;
    }

    // Maximum XOR of any subset (including empty = 0)
    ll max_xor() {
        ll res = 0;
        for (int i = BITS - 1; i >= 0; i--)
            res = max(res, res ^ basis[i]);
        return res;
    }

    // Minimum XOR (excluding 0, assuming basis is non-empty)
    ll min_xor() {
        for (int i = 0; i < BITS; i++)
            if (basis[i]) return basis[i];
        return 0;
    }

    // Number of distinct XOR values = 2^sz (including 0)
    ll count_distinct() { return 1LL << sz; }

    // Reduce to row echelon form for kth smallest query
    vector<ll> reduced_basis() {
        // First reduce
        for (int i = BITS - 1; i >= 0; i--)
            if (basis[i])
                for (int j = i - 1; j >= 0; j--)
                    if ((basis[i] >> j) & 1)
                        basis[i] ^= basis[j];
        // Collect non-zero basis vectors
        vector<ll> rb;
        for (int i = 0; i < BITS; i++)
            if (basis[i]) rb.push_back(basis[i]);
        return rb;
    }

    // Kth smallest XOR value (0-indexed, k=0 gives 0 if include_zero)
    ll kth_smallest(ll k, bool include_zero = true) {
        auto rb = reduced_basis();
        int m = rb.size();
        if (include_zero) {
            if (k == 0) return 0;
            k--; // skip zero
        }
        if (k >= (1LL << m)) return -1; // doesn't exist
        ll res = 0;
        for (int i = 0; i < m; i++)
            if ((k >> i) & 1) res ^= rb[i];
        return res;
    }

    // Merge two bases
    void merge(const XorBasis& other) {
        for (int i = 0; i < BITS; i++)
            if (other.basis[i]) insert(other.basis[i]);
    }
};

// ═══════════════════════════════════════════════════════════
// 7. JOSEPHUS PROBLEM
// ═══════════════════════════════════════════════════════════
/*
 * N people in a circle, every Kth person is eliminated.
 * Find the position of the last survivor (0-indexed).
 */

// O(N) — standard
int josephus(int n, int k) {
    int res = 0;
    for (int i = 2; i <= n; i++)
        res = (res + k) % i;
    return res; // 0-indexed
}

// O(K log N) — for large N, small K
int josephus_fast(int n, int k) {
    if (k == 1) return n - 1;
    if (n == 1) return 0;
    // If k > n, fallback to O(n)
    if (k > n) return josephus(n, k);
    int res = 0;
    for (ll i = 2; i <= n; ) {
        if (res + 1 >= k) {
            // can skip ahead
            ll skip = (i - 1 - res + k - 2) / (k - 1);
            if (skip <= 0) skip = 1;
            if (i + skip - 1 > n) skip = n - i + 1;
            // After adding skip people:
            res = (res + (ll)skip * k) % (i + skip - 1);
            i += skip;
        } else {
            res = (res + k) % i;
            i++;
        }
    }
    return res;
}

// Josephus with k=2 — O(log N) using bit manipulation
int josephus_k2(int n) {
    // Position of highest set bit
    int p = 1;
    while (p * 2 <= n) p *= 2;
    return 2 * (n - p); // 0-indexed: 2*(n-2^m) where 2^m <= n
}

// ═══════════════════════════════════════════════════════════
// 8. IMPARTIAL GAMES FRAMEWORK
// ═══════════════════════════════════════════════════════════
/*
 * For a game that's a SUM of independent sub-games:
 *   1. Compute Grundy number for each sub-game
 *   2. XOR all Grundy numbers
 *   3. If XOR != 0 → first player wins, else second player wins
 *
 * Example: game is played on N independent piles/components
 *   int total_xor = 0;
 *   for (each component c)
 *       total_xor ^= grundy(c);
 *   if (total_xor != 0) → first player wins
 */

// Multi-pile game solver
bool solve_multi_pile_game(const vector<int>& pile_grundy) {
    int xor_sum = 0;
    for (int g : pile_grundy) xor_sum ^= g;
    return xor_sum != 0;
}

// ═══════════════════════════════════════════════════════════
// 9. COMMON GAME PATTERNS
// ═══════════════════════════════════════════════════════════

/*
 * ─────────────────────────────────────
 * SUBTRACTION GAME: pile of N, can take from set S={s1,s2,...,sk}
 *   Grundy(n) follows periodicity of length lcm(max(S)+1, ...)
 *   Compute small table and find period
 *
 * DIVISOR GAME: pile of N, can take d where d | N and d != N
 *   First player wins iff N is even
 *   (For general: compute SG values)
 *
 * TURNING TURTLES / COIN GAMES:
 *   Row of coins, some face-up, some face-down.
 *   Can flip 1-3 coins, rightmost must go up→down.
 *   Grundy(position i) = i (Mock Turtles: Grundy = i+1)
 *   XOR of positions of face-up coins
 *
 * GRAPH GAMES:
 *   Token on a vertex of a DAG. Players take turns moving token.
 *   Grundy(v) = mex(Grundy(u) for all edges v→u)
 *   Multiple tokens: XOR of Grundy values
 *
 * PARTIZAN GAMES (different moves for each player):
 *   Sprague-Grundy does NOT apply
 *   Use surreal number theory (rare in CP)
 * ─────────────────────────────────────
 */

// Subtraction game with explicit computation
bool subtraction_game(int n, const vector<int>& moves) {
    vector<int> g = nim_with_moves(n, moves);
    return g[n] != 0;
}

// Euclid's game: two piles (a, b). Each turn, subtract multiple of smaller from larger.
// First player wins iff:
//   - a >= 2b (can choose to leave (a-b, b) or (a%b, b))
//   - or the game will end in even/odd moves
bool euclid_game(int a, int b) {
    if (a < b) swap(a, b);
    if (b == 0) return false; // current player loses (already lost)
    if (a >= 2 * b) return true; // can control the game
    return !euclid_game(b, a - b); // forced single move
}
