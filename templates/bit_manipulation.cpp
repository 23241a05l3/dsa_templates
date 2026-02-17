/*
 * ============================================================
 *     BIT MANIPULATION — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Basics: set/clear/toggle/check bit
 *    2.  Count set bits (popcount), parity
 *    3.  Power of 2 checks
 *    4.  Lowest / Highest set bit
 *    5.  Iterate over set bits
 *    6.  Enumerate subsets of a bitmask
 *    7.  Iterate all masks with exactly k bits
 *    8.  Bit tricks for CP
 *    9.  XOR properties & tricks
 *   10.  Bitwise Sieve / Bitset tricks
 *   11.  Gray Code
 *   12.  Find missing / duplicate with XOR
 *   13.  Maximum XOR pair (Trie-based)
 *   14.  Bitmask DP utilities
 *   15.  __builtin functions reference
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ═══════════════════════════════════════════════════════════
// 1. BASICS — set / clear / toggle / check
// ═══════════════════════════════════════════════════════════
namespace BitBasics {
    // Set bit at position p (0-indexed from LSB)
    inline int set_bit(int x, int p) { return x | (1 << p); }
    inline ll set_bit(ll x, int p) { return x | (1LL << p); }

    // Clear bit at position p
    inline int clear_bit(int x, int p) { return x & ~(1 << p); }
    inline ll clear_bit(ll x, int p) { return x & ~(1LL << p); }

    // Toggle bit at position p
    inline int toggle_bit(int x, int p) { return x ^ (1 << p); }
    inline ll toggle_bit(ll x, int p) { return x ^ (1LL << p); }

    // Check if bit at position p is set
    inline bool check_bit(int x, int p) { return (x >> p) & 1; }
    inline bool check_bit(ll x, int p) { return (x >> p) & 1; }

    // Set all bits from 0..p-1
    inline int mask_below(int p) { return (1 << p) - 1; }
    inline ll mask_below_ll(int p) { return (1LL << p) - 1; }

    // Clear all bits from position p onwards (keep only lower p bits)
    inline int keep_lower(int x, int p) { return x & ((1 << p) - 1); }
}

// ═══════════════════════════════════════════════════════════
// 2. COUNT SET BITS (popcount) / PARITY
// ═══════════════════════════════════════════════════════════
// Using builtins (fastest)
// __builtin_popcount(x)    — int
// __builtin_popcountll(x)  — long long
// __builtin_parity(x)      — 1 if odd number of set bits, 0 if even

// Manual popcount (Brian Kernighan's)
int popcount_manual(int x) {
    int cnt = 0;
    while (x) { x &= (x - 1); cnt++; }
    return cnt;
}

// Count set bits in range [0, n] for all numbers (dp approach)
vector<int> count_bits_range(int n) {
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) dp[i] = dp[i >> 1] + (i & 1);
    return dp;
}

// Total set bits in all numbers from 1 to n
ll total_set_bits(ll n) {
    if (n <= 0) return 0;
    ll msb = 63 - __builtin_clzll(n); // highest bit position
    ll count = 0;
    for (ll bit = 0; bit <= msb; bit++) {
        ll cycle = 1LL << (bit + 1);
        ll full = (n + 1) / cycle;
        count += full * (1LL << bit);
        ll rem = (n + 1) % cycle;
        count += max(0LL, rem - (1LL << bit));
    }
    return count;
}

// ═══════════════════════════════════════════════════════════
// 3. POWER OF 2 CHECKS
// ═══════════════════════════════════════════════════════════
inline bool is_power_of_2(int x) { return x > 0 && (x & (x - 1)) == 0; }
inline bool is_power_of_2(ll x) { return x > 0 && (x & (x - 1)) == 0; }

// Next power of 2 >= x
int next_power_of_2(int x) {
    if (x <= 1) return 1;
    return 1 << (32 - __builtin_clz(x - 1));
}

// Previous power of 2 <= x
int prev_power_of_2(int x) {
    if (x <= 0) return 0;
    return 1 << (31 - __builtin_clz(x));
}

// ═══════════════════════════════════════════════════════════
// 4. LOWEST / HIGHEST SET BIT
// ═══════════════════════════════════════════════════════════
// Lowest set bit (isolate rightmost 1)
inline int lowest_set_bit(int x) { return x & (-x); }
inline ll lowest_set_bit(ll x) { return x & (-x); }

// Position of lowest set bit (0-indexed)
// __builtin_ctz(x)    — int (count trailing zeros)
// __builtin_ctzll(x)  — long long

// Position of highest set bit (0-indexed)
// 31 - __builtin_clz(x)    — int
// 63 - __builtin_clzll(x)  — long long

// Log2 (floor)
inline int log2_floor(int x) { return 31 - __builtin_clz(x); }
inline int log2_floor(ll x) { return 63 - __builtin_clzll(x); }

// ═══════════════════════════════════════════════════════════
// 5. ITERATE OVER SET BITS
// ═══════════════════════════════════════════════════════════
// Iterate through positions of set bits
void iterate_set_bits(int mask) {
    for (int tmp = mask; tmp; tmp &= (tmp - 1)) {
        int bit = __builtin_ctz(tmp); // position of lowest set bit
        // process bit
    }
}

// Alternative: iterate and remove lowest bit
void iterate_set_bits_v2(int mask) {
    while (mask) {
        int lsb = mask & (-mask); // lowest set bit (as value, not position)
        // process lsb
        mask ^= lsb; // remove it
    }
}

// ═══════════════════════════════════════════════════════════
// 6. ENUMERATE ALL SUBSETS OF A BITMASK
// ═══════════════════════════════════════════════════════════
// All subsets of mask (including 0 and mask itself)
void enumerate_subsets(int mask) {
    for (int sub = mask; ; sub = (sub - 1) & mask) {
        // process sub
        if (sub == 0) break;
    }
    // Total subsets = 2^popcount(mask)
}

// All supersets of mask within universe u
void enumerate_supersets(int mask, int u) {
    for (int sup = mask; sup <= u; sup = (sup + 1) | mask) {
        // process sup
    }
}

// ═══════════════════════════════════════════════════════════
// 7. ITERATE ALL MASKS WITH EXACTLY K BITS SET
// ═══════════════════════════════════════════════════════════
// Gosper's hack: next higher number with same popcount
void iterate_k_bit_masks(int n, int k) {
    int mask = (1 << k) - 1; // smallest mask with k bits
    while (mask < (1 << n)) {
        // process mask
        // Gosper's hack:
        int c = mask & (-mask);
        int r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }
}

// ═══════════════════════════════════════════════════════════
// 8. BIT TRICKS FOR CP
// ═══════════════════════════════════════════════════════════
namespace BitTricks {
    // Swap without temp
    void swap_xor(int& a, int& b) { a ^= b; b ^= a; a ^= b; }

    // Absolute value without branching
    int abs_no_branch(int x) { int mask = x >> 31; return (x + mask) ^ mask; }

    // Min/Max without branching
    int min_no_branch(int a, int b) { return b ^ ((a ^ b) & -(a < b)); }
    int max_no_branch(int a, int b) { return a ^ ((a ^ b) & -(a < b)); }

    // Check if sign differs
    bool signs_differ(int a, int b) { return (a ^ b) < 0; }

    // Average without overflow
    int avg_safe(int a, int b) { return (a & b) + ((a ^ b) >> 1); }

    // Turn off rightmost 1-bit: x & (x-1)
    // Isolate rightmost 1-bit: x & (-x)
    // Right propagate rightmost 1-bit: x | (x-1)
    // Turn on rightmost 0-bit: x | (x+1)

    // Compute x mod 2^n (x must be >= 0)
    int mod_power_of_2(int x, int n) { return x & ((1 << n) - 1); }

    // Check if x is between [lo, hi] (unsigned comparison trick)
    bool in_range(int x, int lo, int hi) {
        return (unsigned)(x - lo) <= (unsigned)(hi - lo);
    }
}

// ═══════════════════════════════════════════════════════════
// 9. XOR PROPERTIES & TRICKS
// ═══════════════════════════════════════════════════════════
/*
 * XOR Properties:
 *   a ^ a = 0
 *   a ^ 0 = a
 *   a ^ b = b ^ a  (commutative)
 *   (a ^ b) ^ c = a ^ (b ^ c) (associative)
 *   a ^ b ^ b = a (self-inverse)
 *
 * XOR from 1 to N:
 *   if N % 4 == 0 → N
 *   if N % 4 == 1 → 1
 *   if N % 4 == 2 → N + 1
 *   if N % 4 == 3 → 0
 */
ll xor_1_to_n(ll n) {
    switch (n & 3) {
        case 0: return n;
        case 1: return 1;
        case 2: return n + 1;
        case 3: return 0;
    }
    return 0;
}

// XOR of range [l, r]
ll xor_range(ll l, ll r) {
    return xor_1_to_n(r) ^ xor_1_to_n(l - 1);
}

// ═══════════════════════════════════════════════════════════
// 10. BITSET TRICKS
// ═══════════════════════════════════════════════════════════
/*
 * bitset<N> bs;
 * bs.count()    — number of set bits
 * bs.any()      — true if any bit set
 * bs.none()     — true if no bit set
 * bs.all()      — true if all bits set
 * bs.set(p)     — set bit p
 * bs.reset(p)   — clear bit p
 * bs.flip(p)    — toggle bit p
 * bs.test(p)    — check bit p
 * bs <<= k      — left shift
 * bs >>= k      — right shift
 * bs &= other   — bitwise AND
 * bs |= other   — bitwise OR
 * bs ^= other   — bitwise XOR
 *
 * COMMON CP USE:
 *   Subset sum with bitset: O(N * MAX / 64)
 *     bitset<MAX> dp; dp[0] = 1;
 *     for (int x : arr) dp |= (dp << x);
 *     // dp[target] tells if target is achievable
 *
 *   Reachability with bitset: O(N^3 / 64)
 *     bitset<N> reach[N];
 *     // Floyd-Warshall style: reach[i] |= reach[j] if reach[i][j]
 */

// ═══════════════════════════════════════════════════════════
// 11. GRAY CODE
// ═══════════════════════════════════════════════════════════
// Binary to Gray: gray(x) = x ^ (x >> 1)
int to_gray(int x) { return x ^ (x >> 1); }

// Gray to Binary
int from_gray(int gray) {
    int num = 0;
    for (; gray; gray >>= 1) num ^= gray;
    return num;
}

// Generate all n-bit Gray codes
vector<int> gray_code(int n) {
    vector<int> res(1 << n);
    for (int i = 0; i < (1 << n); i++) res[i] = i ^ (i >> 1);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 12. FIND MISSING / DUPLICATE WITH XOR
// ═══════════════════════════════════════════════════════════

// Find the single number that appears once (others appear twice)
int find_single(const vector<int>& a) {
    int x = 0;
    for (int v : a) x ^= v;
    return x;
}

// Find two numbers that appear once (others appear twice)
pair<int,int> find_two_singles(const vector<int>& a) {
    int xor_all = 0;
    for (int v : a) xor_all ^= v;
    int diff_bit = xor_all & (-xor_all); // rightmost differing bit
    int g1 = 0, g2 = 0;
    for (int v : a) {
        if (v & diff_bit) g1 ^= v;
        else g2 ^= v;
    }
    return {g1, g2};
}

// Find missing number in [0..n] (one missing)
int find_missing(const vector<int>& a, int n) {
    int x = 0;
    for (int i = 0; i <= n; i++) x ^= i;
    for (int v : a) x ^= v;
    return x;
}

// Find duplicate in [1..n] (one duplicate) — XOR approach
int find_duplicate_xor(const vector<int>& a) {
    int n = a.size() - 1;
    int x = 0;
    for (int i = 1; i <= n; i++) x ^= i;
    for (int v : a) x ^= v;
    return x;
}

// Single Number III: find element appearing once when others appear 3 times
int single_number_three(const vector<int>& a) {
    int ones = 0, twos = 0;
    for (int x : a) {
        ones = (ones ^ x) & ~twos;
        twos = (twos ^ x) & ~ones;
    }
    return ones;
}

// ═══════════════════════════════════════════════════════════
// 13. MAXIMUM XOR PAIR — Trie-based O(N * BITS)
// ═══════════════════════════════════════════════════════════
struct XorTrie {
    static const int BITS = 30;
    struct Node { int ch[2] = {0, 0}; };
    vector<Node> trie;

    XorTrie() { trie.emplace_back(); }

    void insert(int x) {
        int cur = 0;
        for (int i = BITS; i >= 0; i--) {
            int bit = (x >> i) & 1;
            if (!trie[cur].ch[bit]) {
                trie[cur].ch[bit] = trie.size();
                trie.emplace_back();
            }
            cur = trie[cur].ch[bit];
        }
    }

    int max_xor(int x) {
        int cur = 0, res = 0;
        for (int i = BITS; i >= 0; i--) {
            int bit = (x >> i) & 1;
            int want = 1 - bit;
            if (trie[cur].ch[want]) {
                res |= (1 << i);
                cur = trie[cur].ch[want];
            } else {
                cur = trie[cur].ch[bit];
            }
        }
        return res;
    }

    // Find max XOR pair in array
    static int max_xor_pair(const vector<int>& a) {
        XorTrie t;
        int ans = 0;
        for (int x : a) {
            t.insert(x);
            ans = max(ans, t.max_xor(x));
        }
        return ans;
    }
};

// ═══════════════════════════════════════════════════════════
// 14. BITMASK DP UTILITIES
// ═══════════════════════════════════════════════════════════

// Check if subset mask contains element i
inline bool in_mask(int mask, int i) { return (mask >> i) & 1; }

// Add element i to mask
inline int add_to_mask(int mask, int i) { return mask | (1 << i); }

// Remove element i from mask
inline int remove_from_mask(int mask, int i) { return mask & ~(1 << i); }

// Size of subset represented by mask
inline int mask_size(int mask) { return __builtin_popcount(mask); }

// Template: Bitmask DP over subsets
// dp[mask] = min cost to handle elements in mask
// Example: assign n tasks to n workers
vector<int> bitmask_dp_assignment(const vector<vector<int>>& cost) {
    int n = cost.size();
    vector<int> dp(1 << n, INT_MAX);
    dp[0] = 0;
    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == INT_MAX) continue;
        int i = __builtin_popcount(mask); // which worker gets next task
        if (i >= n) continue;
        for (int j = 0; j < n; j++) { // try assigning task j
            if (mask & (1 << j)) continue; // already assigned
            int new_mask = mask | (1 << j);
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[i][j]);
        }
    }
    return dp;
}

// ═══════════════════════════════════════════════════════════
// 15. __builtin FUNCTIONS REFERENCE
// ═══════════════════════════════════════════════════════════
/*
 * __builtin_popcount(x)     — count of set bits (int)
 * __builtin_popcountll(x)   — count of set bits (long long)
 * __builtin_clz(x)          — count leading zeros (undefined for 0)
 * __builtin_clzll(x)        — count leading zeros (long long)
 * __builtin_ctz(x)          — count trailing zeros (undefined for 0)
 * __builtin_ctzll(x)        — count trailing zeros (long long)
 * __builtin_parity(x)       — 1 if odd number of bits set, else 0
 * __builtin_ffs(x)          — position of first set bit (1-indexed), 0 if x=0
 *
 * All are O(1) on modern CPUs.
 *
 * Useful derived:
 *   Highest bit position: 31 - __builtin_clz(x)
 *   Lowest bit position:  __builtin_ctz(x)  (or __builtin_ffs(x) - 1)
 *   Is power of 2:        x > 0 && __builtin_popcount(x) == 1
 *   Floor log2:           31 - __builtin_clz(x)
 *   Ceil log2:            x<=1 ? 0 : 32 - __builtin_clz(x-1)
 */
