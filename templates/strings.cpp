/*
 * ============================================================
 *            STRING ALGORITHMS — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  KMP (Knuth-Morris-Pratt)
 *    2.  Z-Function
 *    3.  Rabin-Karp (Rolling Hash)
 *    4.  Suffix Array + LCP Array — O(N log N)
 *    5.  Aho-Corasick (Multi-pattern matching)
 *    6.  Manacher's Algorithm (Palindromic substrings)
 *    7.  Suffix Automaton (SAM)
 *    8.  Palindromic Tree (Eertree)
 *    9.  String Hashing (Double Hash)
 *   10.  Minimum Rotation (Booth's Algorithm)
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ═══════════════════════════════════════════════════════════
// 1. KMP — O(N + M)
// ═══════════════════════════════════════════════════════════
// Compute failure function (prefix function)
vector<int> kmp_prefix(const string& pattern) {
    int m = pattern.size();
    vector<int> lps(m, 0);
    int len = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

// Find all occurrences of pattern in text
vector<int> kmp_search(const string& text, const string& pattern) {
    int n = text.size(), m = pattern.size();
    vector<int> lps = kmp_prefix(pattern);
    vector<int> matches;
    int i = 0, j = 0;
    while (i < n) {
        if (text[i] == pattern[j]) { i++; j++; }
        if (j == m) {
            matches.push_back(i - m);
            j = lps[j - 1];
        } else if (i < n && text[i] != pattern[j]) {
            if (j) j = lps[j - 1];
            else i++;
        }
    }
    return matches;
}

// ═══════════════════════════════════════════════════════════
// 2. Z-FUNCTION — O(N)
// ═══════════════════════════════════════════════════════════
// z[i] = length of longest substring starting at i that is a prefix of s
vector<int> z_function(const string& s) {
    int n = s.size();
    vector<int> z(n, 0);
    int l = 0, r = 0;
    for (int i = 1; i < n; i++) {
        if (i < r) z[i] = min(r - i, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] > r) { l = i; r = i + z[i]; }
    }
    return z;
}

// Pattern matching using Z-function
vector<int> z_search(const string& text, const string& pattern) {
    string concat = pattern + "$" + text;
    vector<int> z = z_function(concat);
    vector<int> matches;
    int m = pattern.size();
    for (int i = m + 1; i < (int)concat.size(); i++)
        if (z[i] == m) matches.push_back(i - m - 1);
    return matches;
}

// ═══════════════════════════════════════════════════════════
// 3. RABIN-KARP (Rolling Hash) — O(N + M) expected
// ═══════════════════════════════════════════════════════════
struct RollingHash {
    static const ll BASE1 = 131, BASE2 = 137;
    static const ll MOD1 = 1e9 + 7, MOD2 = 1e9 + 9;
    vector<ll> h1, h2, pw1, pw2;
    int n;

    RollingHash(const string& s) : n(s.size()), h1(n+1), h2(n+1), pw1(n+1), pw2(n+1) {
        pw1[0] = pw2[0] = 1;
        h1[0] = h2[0] = 0;
        for (int i = 0; i < n; i++) {
            h1[i+1] = (h1[i] * BASE1 + s[i]) % MOD1;
            h2[i+1] = (h2[i] * BASE2 + s[i]) % MOD2;
            pw1[i+1] = pw1[i] * BASE1 % MOD1;
            pw2[i+1] = pw2[i] * BASE2 % MOD2;
        }
    }

    // Hash of substring [l, r] (0-indexed, inclusive)
    pair<ll,ll> get_hash(int l, int r) {
        ll v1 = (h1[r+1] - h1[l] * pw1[r-l+1] % MOD1 + MOD1 * 2) % MOD1;
        ll v2 = (h2[r+1] - h2[l] * pw2[r-l+1] % MOD2 + MOD2 * 2) % MOD2;
        return {v1, v2};
    }

    bool equal(int l1, int r1, int l2, int r2) {
        return get_hash(l1, r1) == get_hash(l2, r2);
    }

    // LCP of substrings starting at i and j (binary search)
    int lcp(int i, int j) {
        int lo = 0, hi = min(n - i, n - j);
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (get_hash(i, i+mid-1) == get_hash(j, j+mid-1)) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
};

// ═══════════════════════════════════════════════════════════
// 4. SUFFIX ARRAY + LCP ARRAY — O(N log N)
// ═══════════════════════════════════════════════════════════
struct SuffixArray {
    int n;
    string s;
    vector<int> sa, rank_, lcp;

    SuffixArray(const string& s) : s(s), n(s.size()) { build(); build_lcp(); }

    void build() {
        sa.resize(n); rank_.resize(n);
        iota(sa.begin(), sa.end(), 0);
        for (int i = 0; i < n; i++) rank_[i] = s[i];

        for (int gap = 1; gap < n; gap <<= 1) {
            auto cmp = [&](int a, int b) {
                if (rank_[a] != rank_[b]) return rank_[a] < rank_[b];
                int ra = a + gap < n ? rank_[a + gap] : -1;
                int rb = b + gap < n ? rank_[b + gap] : -1;
                return ra < rb;
            };
            sort(sa.begin(), sa.end(), cmp);
            vector<int> tmp(n);
            tmp[sa[0]] = 0;
            for (int i = 1; i < n; i++)
                tmp[sa[i]] = tmp[sa[i-1]] + cmp(sa[i-1], sa[i]);
            rank_ = tmp;
            if (rank_[sa[n-1]] == n - 1) break;
        }
    }

    void build_lcp() {
        lcp.resize(n, 0);
        vector<int> inv(n);
        for (int i = 0; i < n; i++) inv[sa[i]] = i;
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (inv[i] == 0) { k = 0; continue; }
            int j = sa[inv[i] - 1];
            while (i + k < n && j + k < n && s[i+k] == s[j+k]) k++;
            lcp[inv[i]] = k;
            if (k) k--;
        }
    }

    // Count distinct substrings = n*(n+1)/2 - sum(lcp)
    ll count_distinct() {
        ll total = (ll)n * (n + 1) / 2;
        for (int i = 1; i < n; i++) total -= lcp[i];
        return total;
    }
};

// ═══════════════════════════════════════════════════════════
// 5. AHO-CORASICK — Multi-pattern matching — O(N + M + Z)
// ═══════════════════════════════════════════════════════════
struct AhoCorasick {
    struct Node {
        int children[26];
        int fail;        // failure link
        int output;      // output link (pattern end)
        int pattern_id;  // which pattern ends here (-1 if none)
        int depth;
    };
    vector<Node> nodes;

    AhoCorasick() {
        nodes.push_back(Node());
        memset(nodes[0].children, -1, sizeof(nodes[0].children));
        nodes[0].fail = 0;
        nodes[0].output = -1;
        nodes[0].pattern_id = -1;
        nodes[0].depth = 0;
    }

    int new_node() {
        nodes.push_back(Node());
        memset(nodes.back().children, -1, sizeof(nodes.back().children));
        nodes.back().fail = 0;
        nodes.back().output = -1;
        nodes.back().pattern_id = -1;
        nodes.back().depth = 0;
        return nodes.size() - 1;
    }

    void add_pattern(const string& s, int id) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (nodes[cur].children[idx] == -1)
                nodes[cur].children[idx] = new_node();
            cur = nodes[cur].children[idx];
            nodes[cur].depth = nodes[nodes[cur].fail].depth + 1; // approximate
        }
        nodes[cur].pattern_id = id;
    }

    void build() {
        queue<int> q;
        for (int c = 0; c < 26; c++) {
            if (nodes[0].children[c] == -1)
                nodes[0].children[c] = 0; // point to root
            else {
                nodes[nodes[0].children[c]].fail = 0;
                q.push(nodes[0].children[c]);
            }
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            nodes[u].output = (nodes[nodes[u].fail].pattern_id >= 0) ?
                              nodes[u].fail : nodes[nodes[u].fail].output;
            for (int c = 0; c < 26; c++) {
                if (nodes[u].children[c] == -1)
                    nodes[u].children[c] = nodes[nodes[u].fail].children[c];
                else {
                    nodes[nodes[u].children[c]].fail = nodes[nodes[u].fail].children[c];
                    q.push(nodes[u].children[c]);
                }
            }
        }
    }

    // Search text for all patterns. Returns vector of {position, pattern_id}
    vector<pair<int,int>> search(const string& text) {
        vector<pair<int,int>> results;
        int cur = 0;
        for (int i = 0; i < (int)text.size(); i++) {
            cur = nodes[cur].children[text[i] - 'a'];
            // Check all patterns ending here
            int tmp = cur;
            while (tmp > 0) {
                if (nodes[tmp].pattern_id >= 0)
                    results.push_back({i, nodes[tmp].pattern_id});
                tmp = nodes[tmp].output;
            }
        }
        return results;
    }
};

// ═══════════════════════════════════════════════════════════
// 6. MANACHER'S ALGORITHM — All palindromic substrings O(N)
// ═══════════════════════════════════════════════════════════
struct Manacher {
    // d1[i] = radius of longest odd-length palindrome centered at i
    // d2[i] = radius of longest even-length palindrome between i-1 and i
    vector<int> d1, d2;

    Manacher(const string& s) {
        int n = s.size();
        d1.resize(n); d2.resize(n);

        // Odd-length palindromes
        for (int i = 0, l = 0, r = -1; i < n; i++) {
            int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
            while (i - k >= 0 && i + k < n && s[i-k] == s[i+k]) k++;
            d1[i] = k--;
            if (i + k > r) { l = i - k; r = i + k; }
        }

        // Even-length palindromes
        for (int i = 0, l = 0, r = -1; i < n; i++) {
            int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
            while (i - k - 1 >= 0 && i + k < n && s[i-k-1] == s[i+k]) k++;
            d2[i] = k--;
            if (i + k > r) { l = i - k - 1; r = i + k; }
        }
    }

    // Check if s[l..r] is a palindrome
    bool is_palindrome(int l, int r) {
        int len = r - l + 1;
        int mid = (l + r) / 2;
        if (len & 1) return d1[mid] >= len / 2 + 1;
        return d2[mid] >= len / 2;
    }

    // Count total palindromic substrings
    ll count() {
        ll cnt = 0;
        for (int i = 0; i < (int)d1.size(); i++) cnt += d1[i]; // odd
        for (int i = 0; i < (int)d2.size(); i++) cnt += d2[i]; // even
        return cnt;
    }

    // Longest palindromic substring
    pair<int,int> longest() {
        int best = 0, center = 0;
        bool odd = true;
        for (int i = 0; i < (int)d1.size(); i++)
            if (2 * d1[i] - 1 > best) { best = 2 * d1[i] - 1; center = i; odd = true; }
        for (int i = 0; i < (int)d2.size(); i++)
            if (2 * d2[i] > best) { best = 2 * d2[i]; center = i; odd = false; }
        int l = odd ? center - d1[center] + 1 : center - d2[center];
        return {l, l + best - 1};
    }
};

// ═══════════════════════════════════════════════════════════
// 7. SUFFIX AUTOMATON (SAM) — O(N)
// ═══════════════════════════════════════════════════════════
struct SuffixAutomaton {
    struct State {
        int len, link;
        map<char, int> next;
        ll cnt;  // number of times state's endpos set has elements
    };
    vector<State> st;
    int last;

    SuffixAutomaton() {
        st.push_back({0, -1, {}, 0});
        last = 0;
    }

    void extend(char c) {
        int cur = st.size();
        st.push_back({st[last].len + 1, -1, {}, 1});
        int p = last;
        while (p != -1 && !st[p].next.count(c)) {
            st[p].next[c] = cur;
            p = st[p].link;
        }
        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = st.size();
                st.push_back({st[p].len + 1, st[q].link, st[q].next, 0});
                while (p != -1 && st[p].next[c] == q) {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }
                st[q].link = clone;
                st[cur].link = clone;
            }
        }
        last = cur;
    }

    void build(const string& s) {
        for (char c : s) extend(c);
    }

    // Count distinct substrings
    ll count_distinct() {
        ll cnt = 0;
        for (int i = 1; i < (int)st.size(); i++)
            cnt += st[i].len - st[st[i].link].len;
        return cnt;
    }
};

// ═══════════════════════════════════════════════════════════
// 8. PALINDROMIC TREE (Eertree)
// ═══════════════════════════════════════════════════════════
struct PalindromicTree {
    struct Node {
        int len, link;
        map<char, int> children;
        int cnt; // number of times this palindrome appears
    };
    vector<Node> nodes;
    string s;
    int last; // last added node

    PalindromicTree() {
        nodes.push_back({-1, 0, {}, 0}); // imaginary root (len=-1)
        nodes.push_back({0, 0, {}, 0});   // empty string root
        last = 1;
    }

    int get_link(int v, int i) {
        while (i - 1 - nodes[v].len < 0 || s[i - 1 - nodes[v].len] != s[i])
            v = nodes[v].link;
        return v;
    }

    void add(char c) {
        s += c;
        int i = s.size() - 1;
        int cur = get_link(last, i);
        if (nodes[cur].children.count(c)) {
            last = nodes[cur].children[c];
            nodes[last].cnt++;
            return;
        }
        int now = nodes.size();
        nodes.push_back({nodes[cur].len + 2, -1, {}, 1});
        nodes[cur].children[c] = now;
        if (nodes[now].len == 1) {
            nodes[now].link = 1; // link to empty string
        } else {
            nodes[now].link = nodes[get_link(nodes[cur].link, i)].children[c];
        }
        last = now;
    }

    // Propagate counts and count total palindromic substrings
    ll count_all() {
        ll total = 0;
        for (int i = nodes.size() - 1; i >= 2; i--) {
            nodes[nodes[i].link].cnt += nodes[i].cnt;
            total += nodes[i].cnt;
        }
        return total;
    }

    int distinct_palindromes() { return nodes.size() - 2; }
};

// ═══════════════════════════════════════════════════════════
// 9. STRING HASHING UTILITIES
// ═══════════════════════════════════════════════════════════

// Quick hash for a string (good for hash maps)
struct StringHash {
    size_t operator()(const string& s) const {
        size_t h = 0;
        for (char c : s) h = h * 131 + c;
        return h;
    }
};

// Compare two substrings across different strings using hashing
// (Use two RollingHash objects and compare their get_hash results)

// ═══════════════════════════════════════════════════════════
// 10. MINIMUM ROTATION — Booth's Algorithm — O(N)
// ═══════════════════════════════════════════════════════════
// Returns starting index of lexicographically smallest rotation
int min_rotation(const string& s) {
    int n = s.size();
    string ss = s + s;
    vector<int> f(2 * n, -1);
    int k = 0;
    for (int j = 1; j < 2 * n; j++) {
        int i = f[j - 1 - k];
        while (i != -1 && ss[j] != ss[k + i + 1]) {
            if (ss[j] < ss[k + i + 1]) k = j - i - 1;
            i = f[i];
        }
        if (i == -1 && ss[j] != ss[k + i + 1]) {
            if (ss[j] < ss[k + i + 1]) k = j;
            f[j - k] = -1;
        } else {
            f[j - k] = i + 1;
        }
    }
    return k;
}

/*
 * ══════════════════════════════════════
 *  STRING ALGORITHM SELECTION GUIDE
 * ══════════════════════════════════════
 *
 * Single pattern search:
 *   - KMP or Z-function (both O(N+M))
 *   - Rabin-Karp for multiple hash comparisons
 *
 * Multiple pattern search:
 *   - Aho-Corasick — search all patterns simultaneously
 *
 * Palindromes:
 *   - Manacher's — all palindromic substrings in O(N)
 *   - Palindromic Tree — distinct palindromes + counts
 *
 * Suffix structures:
 *   - Suffix Array — sorted suffixes, LCP, count distinct substrings
 *   - Suffix Automaton — powerful for substring queries
 *
 * Substring comparison / LCP:
 *   - Rolling Hash — O(1) compare after O(N) build
 *   - Suffix Array + LCP + sparse table
 *
 * Lexicographic min rotation:
 *   - Booth's algorithm — O(N)
 *
 * Common tricks:
 *   - Concatenate pattern + "$" + text for Z/KMP matching
 *   - Reverse string problems → often suffix of reverse = prefix
 *   - Hashing: always use double hash to avoid collisions
 */
