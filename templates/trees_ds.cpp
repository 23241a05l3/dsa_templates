/*
 * ============================================================
 *       TREES & ADVANCED DATA STRUCTURES — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Segment Tree (Point update, Range query)
 *    2.  Segment Tree with Lazy Propagation
 *    3.  Persistent Segment Tree
 *    4.  Merge Sort Tree (for Kth smallest in range)
 *    5.  Fenwick Tree / Binary Indexed Tree (BIT)
 *    6.  2D Fenwick Tree
 *    7.  Sparse Table (RMQ) — O(1) query
 *    8.  Trie (Binary & String)
 *    9.  DSU (Union-Find) with Rollback
 *   10.  Heavy-Light Decomposition (HLD)
 *   11.  Centroid Decomposition
 *   12.  Treap (Implicit — acts as balanced BST / array)
 *   13.  Ordered Set (Policy-based)
 *   14.  Monotone Stack / Monotone Deque
 *   15.  Linked List (for CP)
 *   16.  Min/Max Heap utilities
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = 1e18;
const int MOD = 1e9 + 7;

// ═══════════════════════════════════════════════════════════
// 1. SEGMENT TREE — Point Update, Range Query — O(log N)
// ═══════════════════════════════════════════════════════════
struct SegTree {
    int n;
    vector<ll> tree;

    SegTree() {}
    SegTree(int n) : n(n), tree(4 * n, 0) {}
    SegTree(const vector<ll>& a) : n(a.size()), tree(4 * a.size()) { build(a, 1, 0, n - 1); }

    void build(const vector<ll>& a, int node, int lo, int hi) {
        if (lo == hi) { tree[node] = a[lo]; return; }
        int mid = (lo + hi) / 2;
        build(a, 2*node, lo, mid);
        build(a, 2*node+1, mid+1, hi);
        tree[node] = tree[2*node] + tree[2*node+1]; // change for min/max/gcd
    }

    void update(int pos, ll val, int node, int lo, int hi) {
        if (lo == hi) { tree[node] = val; return; }
        int mid = (lo + hi) / 2;
        if (pos <= mid) update(pos, val, 2*node, lo, mid);
        else update(pos, val, 2*node+1, mid+1, hi);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
    void update(int pos, ll val) { update(pos, val, 1, 0, n - 1); }

    ll query(int l, int r, int node, int lo, int hi) {
        if (r < lo || hi < l) return 0; // identity for sum; INF for min; -INF for max
        if (l <= lo && hi <= r) return tree[node];
        int mid = (lo + hi) / 2;
        return query(l, r, 2*node, lo, mid) + query(l, r, 2*node+1, mid+1, hi);
    }
    ll query(int l, int r) { return query(l, r, 1, 0, n - 1); }

    // Find first index >= l where prefix sum >= val (binary search on segtree)
    int walk(int l, ll val, int node, int lo, int hi) {
        if (hi < l || tree[node] < val) return -1;
        if (lo == hi) return lo;
        int mid = (lo + hi) / 2;
        int res = walk(l, val, 2*node, lo, mid);
        if (res != -1) return res;
        return walk(l, val - tree[2*node], 2*node+1, mid+1, hi);
    }
};

// ═══════════════════════════════════════════════════════════
// 2. SEGMENT TREE + LAZY PROPAGATION — Range Update, Range Query
// ═══════════════════════════════════════════════════════════
struct LazySegTree {
    int n;
    vector<ll> tree, lazy;

    LazySegTree() {}
    LazySegTree(int n) : n(n), tree(4*n, 0), lazy(4*n, 0) {}
    LazySegTree(const vector<ll>& a) : n(a.size()), tree(4*a.size()), lazy(4*a.size(), 0) {
        build(a, 1, 0, n-1);
    }

    void build(const vector<ll>& a, int node, int lo, int hi) {
        if (lo == hi) { tree[node] = a[lo]; return; }
        int mid = (lo + hi) / 2;
        build(a, 2*node, lo, mid);
        build(a, 2*node+1, mid+1, hi);
        tree[node] = tree[2*node] + tree[2*node+1];
    }

    void push_down(int node, int lo, int hi) {
        if (lazy[node] == 0) return;
        int mid = (lo + hi) / 2;
        apply_lazy(2*node, lo, mid, lazy[node]);
        apply_lazy(2*node+1, mid+1, hi, lazy[node]);
        lazy[node] = 0;
    }

    void apply_lazy(int node, int lo, int hi, ll val) {
        tree[node] += val * (hi - lo + 1);  // For range add + range sum
        lazy[node] += val;
    }

    // Range add: add val to all elements in [l, r]
    void update(int l, int r, ll val, int node, int lo, int hi) {
        if (r < lo || hi < l) return;
        if (l <= lo && hi <= r) { apply_lazy(node, lo, hi, val); return; }
        push_down(node, lo, hi);
        int mid = (lo + hi) / 2;
        update(l, r, val, 2*node, lo, mid);
        update(l, r, val, 2*node+1, mid+1, hi);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
    void update(int l, int r, ll val) { update(l, r, val, 1, 0, n-1); }

    // Range set: set all elements in [l, r] to val
    // (change apply_lazy: tree[node] = val * (hi-lo+1), lazy[node] = val, use sentinel)

    ll query(int l, int r, int node, int lo, int hi) {
        if (r < lo || hi < l) return 0;
        if (l <= lo && hi <= r) return tree[node];
        push_down(node, lo, hi);
        int mid = (lo + hi) / 2;
        return query(l, r, 2*node, lo, mid) + query(l, r, 2*node+1, mid+1, hi);
    }
    ll query(int l, int r) { return query(l, r, 1, 0, n-1); }
};

// ═══════════════════════════════════════════════════════════
// 3. PERSISTENT SEGMENT TREE — O(log N) per version
// ═══════════════════════════════════════════════════════════
struct PersistentSegTree {
    struct Node {
        ll val;
        int left, right;
    };
    vector<Node> nodes;
    vector<int> roots;
    int n;

    PersistentSegTree(int n) : n(n) {
        nodes.reserve(20 * n); // preallocate
        roots.push_back(build(0, n - 1));
    }

    int new_node(ll val = 0, int l = 0, int r = 0) {
        nodes.push_back({val, l, r});
        return nodes.size() - 1;
    }

    int build(int lo, int hi) {
        if (lo == hi) return new_node(0);
        int mid = (lo + hi) / 2;
        int l = build(lo, mid), r = build(mid + 1, hi);
        return new_node(nodes[l].val + nodes[r].val, l, r);
    }

    int update(int prev, int pos, ll val, int lo, int hi) {
        if (lo == hi) return new_node(nodes[prev].val + val);
        int mid = (lo + hi) / 2;
        int l = nodes[prev].left, r = nodes[prev].right;
        if (pos <= mid) l = update(l, pos, val, lo, mid);
        else r = update(r, pos, val, mid + 1, hi);
        return new_node(nodes[l].val + nodes[r].val, l, r);
    }

    // Create new version with pos updated
    void update(int pos, ll val) {
        roots.push_back(update(roots.back(), pos, val, 0, n - 1));
    }

    ll query(int node, int l, int r, int lo, int hi) {
        if (r < lo || hi < l) return 0;
        if (l <= lo && hi <= r) return nodes[node].val;
        int mid = (lo + hi) / 2;
        return query(nodes[node].left, l, r, lo, mid) +
               query(nodes[node].right, l, r, mid + 1, hi);
    }

    // Query version v
    ll query(int version, int l, int r) {
        return query(roots[version], l, r, 0, n - 1);
    }

    // Kth smallest in range [l, r] using persistent segtree on sorted values
    int kth_smallest(int lv, int rv, int k, int lo, int hi) {
        if (lo == hi) return lo;
        int mid = (lo + hi) / 2;
        int cnt = nodes[nodes[rv].left].val - nodes[nodes[lv].left].val;
        if (cnt >= k) return kth_smallest(nodes[lv].left, nodes[rv].left, k, lo, mid);
        return kth_smallest(nodes[lv].right, nodes[rv].right, k - cnt, mid + 1, hi);
    }
};

// ═══════════════════════════════════════════════════════════
// 4. MERGE SORT TREE — O(N log² N) for count in range
// ═══════════════════════════════════════════════════════════
struct MergeSortTree {
    int n;
    vector<vector<int>> tree;

    MergeSortTree(const vector<int>& a) : n(a.size()), tree(4 * a.size()) {
        build(a, 1, 0, n - 1);
    }

    void build(const vector<int>& a, int node, int lo, int hi) {
        if (lo == hi) { tree[node] = {a[lo]}; return; }
        int mid = (lo + hi) / 2;
        build(a, 2*node, lo, mid);
        build(a, 2*node+1, mid+1, hi);
        merge(tree[2*node].begin(), tree[2*node].end(),
              tree[2*node+1].begin(), tree[2*node+1].end(),
              back_inserter(tree[node]));
    }

    // Count of elements <= val in range [l, r]
    int query(int l, int r, int val, int node, int lo, int hi) {
        if (r < lo || hi < l) return 0;
        if (l <= lo && hi <= r)
            return upper_bound(tree[node].begin(), tree[node].end(), val) - tree[node].begin();
        int mid = (lo + hi) / 2;
        return query(l, r, val, 2*node, lo, mid) + query(l, r, val, 2*node+1, mid+1, hi);
    }

    int query(int l, int r, int val) { return query(l, r, val, 1, 0, n - 1); }
};

// ═══════════════════════════════════════════════════════════
// 5. FENWICK TREE (BIT) — O(log N)
// ═══════════════════════════════════════════════════════════
struct BIT {
    int n;
    vector<ll> tree;

    BIT() {}
    BIT(int n) : n(n), tree(n + 1, 0) {}

    void update(int i, ll delta) { // 1-indexed
        for (; i <= n; i += i & (-i)) tree[i] += delta;
    }

    ll query(int i) { // prefix sum [1..i]
        ll sum = 0;
        for (; i > 0; i -= i & (-i)) sum += tree[i];
        return sum;
    }

    ll query(int l, int r) { return query(r) - query(l - 1); }

    // Find smallest index with prefix sum >= val
    int kth(ll val) {
        int pos = 0;
        for (int pw = 1 << __lg(n); pw > 0; pw >>= 1) {
            if (pos + pw <= n && tree[pos + pw] < val) {
                pos += pw;
                val -= tree[pos];
            }
        }
        return pos + 1;
    }
};

// Range update, Point query BIT (using difference array)
struct BIT_RangeUpdate {
    BIT bit;
    BIT_RangeUpdate(int n) : bit(n) {}
    void update(int l, int r, ll val) { bit.update(l, val); bit.update(r + 1, -val); }
    ll query(int i) { return bit.query(i); }
};

// Range update, Range query BIT (two BITs)
struct BIT_RangeUpdateRangeQuery {
    BIT b1, b2;
    int n;
    BIT_RangeUpdateRangeQuery(int n) : n(n), b1(n), b2(n) {}
    void update(int l, int r, ll val) {
        b1.update(l, val); b1.update(r+1, -val);
        b2.update(l, val * (l-1)); b2.update(r+1, -val * r);
    }
    ll prefix(int i) { return b1.query(i) * i - b2.query(i); }
    ll query(int l, int r) { return prefix(r) - prefix(l-1); }
};

// ═══════════════════════════════════════════════════════════
// 6. 2D FENWICK TREE — O(log N * log M)
// ═══════════════════════════════════════════════════════════
struct BIT2D {
    int n, m;
    vector<vector<ll>> tree;

    BIT2D(int n, int m) : n(n), m(m), tree(n + 1, vector<ll>(m + 1, 0)) {}

    void update(int x, int y, ll val) {
        for (int i = x; i <= n; i += i & (-i))
            for (int j = y; j <= m; j += j & (-j))
                tree[i][j] += val;
    }

    ll query(int x, int y) {
        ll sum = 0;
        for (int i = x; i > 0; i -= i & (-i))
            for (int j = y; j > 0; j -= j & (-j))
                sum += tree[i][j];
        return sum;
    }

    ll query(int x1, int y1, int x2, int y2) {
        return query(x2, y2) - query(x1-1, y2) - query(x2, y1-1) + query(x1-1, y1-1);
    }
};

// ═══════════════════════════════════════════════════════════
// 7. SPARSE TABLE — O(N log N) build, O(1) query (RMQ)
// ═══════════════════════════════════════════════════════════
struct SparseTable {
    int n, LOG;
    vector<vector<int>> table; // stores indices for min
    vector<int> a;

    SparseTable() {}
    SparseTable(const vector<int>& a) : a(a), n(a.size()) {
        LOG = __lg(n) + 1;
        table.assign(LOG, vector<int>(n));
        for (int i = 0; i < n; i++) table[0][i] = i;
        for (int k = 1; k < LOG; k++)
            for (int i = 0; i + (1 << k) <= n; i++) {
                int l = table[k-1][i], r = table[k-1][i + (1 << (k-1))];
                table[k][i] = (a[l] <= a[r]) ? l : r;
            }
    }

    int query_idx(int l, int r) { // returns INDEX of min
        int k = __lg(r - l + 1);
        int li = table[k][l], ri = table[k][r - (1 << k) + 1];
        return (a[li] <= a[ri]) ? li : ri;
    }

    int query(int l, int r) { return a[query_idx(l, r)]; }
};

// ═══════════════════════════════════════════════════════════
// 8. TRIE — String Trie + Binary Trie (XOR)
// ═══════════════════════════════════════════════════════════

// String Trie — insert/search/prefix count
struct Trie {
    struct Node {
        int children[26];
        int cnt = 0;     // count of strings passing through
        int end_cnt = 0; // count of strings ending here
    };
    vector<Node> nodes;

    Trie() { nodes.push_back(Node()); memset(nodes[0].children, -1, sizeof(nodes[0].children)); }

    int new_node() {
        nodes.push_back(Node());
        memset(nodes.back().children, -1, sizeof(nodes.back().children));
        return nodes.size() - 1;
    }

    void insert(const string& s) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (nodes[cur].children[idx] == -1)
                nodes[cur].children[idx] = new_node();
            cur = nodes[cur].children[idx];
            nodes[cur].cnt++;
        }
        nodes[cur].end_cnt++;
    }

    bool search(const string& s) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (nodes[cur].children[idx] == -1) return false;
            cur = nodes[cur].children[idx];
        }
        return nodes[cur].end_cnt > 0;
    }

    int count_prefix(const string& prefix) {
        int cur = 0;
        for (char c : prefix) {
            int idx = c - 'a';
            if (nodes[cur].children[idx] == -1) return 0;
            cur = nodes[cur].children[idx];
        }
        return nodes[cur].cnt;
    }

    void erase(const string& s) { // assumes s exists
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            cur = nodes[cur].children[idx];
            nodes[cur].cnt--;
        }
        nodes[cur].end_cnt--;
    }
};

// Binary Trie — for maximum XOR queries
struct BinaryTrie {
    static const int MAXBIT = 30; // adjust for problem (30 for int, 60 for long long)
    struct Node {
        int children[2];
        int cnt;
    };
    vector<Node> nodes;

    BinaryTrie() { nodes.push_back({{-1, -1}, 0}); }

    int new_node() { nodes.push_back({{-1, -1}, 0}); return nodes.size() - 1; }

    void insert(int x) {
        int cur = 0;
        for (int i = MAXBIT; i >= 0; i--) {
            int bit = (x >> i) & 1;
            if (nodes[cur].children[bit] == -1)
                nodes[cur].children[bit] = new_node();
            cur = nodes[cur].children[bit];
            nodes[cur].cnt++;
        }
    }

    void erase(int x) {
        int cur = 0;
        for (int i = MAXBIT; i >= 0; i--) {
            int bit = (x >> i) & 1;
            cur = nodes[cur].children[bit];
            nodes[cur].cnt--;
        }
    }

    // Find max XOR of x with any number in trie
    int max_xor(int x) {
        int cur = 0, result = 0;
        for (int i = MAXBIT; i >= 0; i--) {
            int bit = (x >> i) & 1;
            int want = 1 - bit; // prefer opposite bit
            if (nodes[cur].children[want] != -1 && nodes[nodes[cur].children[want]].cnt > 0) {
                result |= (1 << i);
                cur = nodes[cur].children[want];
            } else {
                cur = nodes[cur].children[bit];
            }
        }
        return result;
    }
};

// ═══════════════════════════════════════════════════════════
// 9. DSU WITH ROLLBACK (for offline divide & conquer)
// ═══════════════════════════════════════════════════════════
struct DSU_Rollback {
    vector<int> par, rnk;
    vector<pair<int*,int>> history; // for rollback
    int components;

    DSU_Rollback(int n) : par(n), rnk(n, 0), components(n) {
        iota(par.begin(), par.end(), 0);
    }

    int find(int x) { // no path compression (for rollback support)
        while (par[x] != x) x = par[x];
        return x;
    }

    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rnk[a] < rnk[b]) swap(a, b);
        history.push_back({&par[b], par[b]});
        history.push_back({&rnk[a], rnk[a]});
        par[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
        components--;
        return true;
    }

    int save() { return history.size(); }

    void rollback(int checkpoint) {
        while ((int)history.size() > checkpoint) {
            *history.back().first = history.back().second;
            history.pop_back();
            // Note: components tracking needs manual rollback if used
        }
    }
};

// ═══════════════════════════════════════════════════════════
// 10. HEAVY-LIGHT DECOMPOSITION (HLD) — O(log² N) per query
// ═══════════════════════════════════════════════════════════
struct HLD {
    int n;
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos, sz;
    int cur_pos;
    LazySegTree seg; // or SegTree (from above)

    HLD(int n) : n(n), adj(n), parent(n), depth(n), heavy(n, -1),
                 head(n), pos(n), sz(n), cur_pos(0) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int dfs_size(int u, int p, int d) {
        parent[u] = p; depth[u] = d; sz[u] = 1;
        int max_sz = 0;
        for (int v : adj[u]) {
            if (v == p) continue;
            sz[u] += dfs_size(v, u, d + 1);
            if (sz[v] > max_sz) { max_sz = sz[v]; heavy[u] = v; }
        }
        return sz[u];
    }

    void decompose(int u, int h) {
        head[u] = h; pos[u] = cur_pos++;
        if (heavy[u] != -1) decompose(heavy[u], h);
        for (int v : adj[u])
            if (v != parent[u] && v != heavy[u]) decompose(v, v);
    }

    void build(const vector<ll>& values) {
        dfs_size(0, -1, 0);
        decompose(0, 0);
        vector<ll> seg_vals(n);
        for (int i = 0; i < n; i++) seg_vals[pos[i]] = values[i];
        seg = LazySegTree(seg_vals);
    }

    // Path query (sum on path u -> v)
    ll path_query(int u, int v) {
        ll res = 0;
        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]]) swap(u, v);
            res += seg.query(pos[head[u]], pos[u]);
            u = parent[head[u]];
        }
        if (depth[u] > depth[v]) swap(u, v);
        res += seg.query(pos[u], pos[v]);
        return res;
    }

    // Path update (add val on path u -> v)
    void path_update(int u, int v, ll val) {
        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]]) swap(u, v);
            seg.update(pos[head[u]], pos[u], val);
            u = parent[head[u]];
        }
        if (depth[u] > depth[v]) swap(u, v);
        seg.update(pos[u], pos[v], val);
    }

    // Subtree query
    ll subtree_query(int u) { return seg.query(pos[u], pos[u] + sz[u] - 1); }
    void subtree_update(int u, ll val) { seg.update(pos[u], pos[u] + sz[u] - 1, val); }
};

// ═══════════════════════════════════════════════════════════
// 11. CENTROID DECOMPOSITION — O(N log N)
// ═══════════════════════════════════════════════════════════
struct CentroidDecomp {
    int n;
    vector<vector<int>> adj;
    vector<int> sz, cpar;
    vector<bool> removed;

    CentroidDecomp(int n) : n(n), adj(n), sz(n), cpar(n, -1), removed(n, false) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int get_size(int u, int p) {
        sz[u] = 1;
        for (int v : adj[u])
            if (v != p && !removed[v]) sz[u] += get_size(v, u);
        return sz[u];
    }

    int get_centroid(int u, int p, int tree_sz) {
        for (int v : adj[u])
            if (v != p && !removed[v] && sz[v] > tree_sz / 2)
                return get_centroid(v, u, tree_sz);
        return u;
    }

    void build(int u = 0, int p = -1) {
        int tree_sz = get_size(u, -1);
        int centroid = get_centroid(u, -1, tree_sz);
        cpar[centroid] = p;
        removed[centroid] = true;

        for (int v : adj[centroid])
            if (!removed[v]) build(v, centroid);
    }
    // After build: cpar[v] = parent in centroid tree
    // Process queries by climbing centroid tree: O(log N) ancestors
};

// ═══════════════════════════════════════════════════════════
// 12. TREAP (Implicit) — Balanced BST as array, O(log N) split/merge
// ═══════════════════════════════════════════════════════════
struct Treap {
    struct Node {
        ll val, sum;
        int sz, pri;
        bool rev; // lazy reverse flag
        int left, right;
    };
    vector<Node> nodes;

    Treap() { nodes.push_back({0, 0, 0, 0, false, 0, 0}); } // dummy node 0

    int new_node(ll val) {
        nodes.push_back({val, val, 1, rand(), false, 0, 0});
        return nodes.size() - 1;
    }

    int sz(int t) { return t ? nodes[t].sz : 0; }
    ll sum(int t) { return t ? nodes[t].sum : 0; }

    void pull(int t) {
        if (!t) return;
        nodes[t].sz = 1 + sz(nodes[t].left) + sz(nodes[t].right);
        nodes[t].sum = nodes[t].val + sum(nodes[t].left) + sum(nodes[t].right);
    }

    void push(int t) {
        if (!t || !nodes[t].rev) return;
        swap(nodes[t].left, nodes[t].right);
        if (nodes[t].left) nodes[nodes[t].left].rev ^= 1;
        if (nodes[t].right) nodes[nodes[t].right].rev ^= 1;
        nodes[t].rev = false;
    }

    // Split: first k elements go to l, rest to r
    void split(int t, int k, int& l, int& r) {
        if (!t) { l = r = 0; return; }
        push(t);
        int left_sz = sz(nodes[t].left);
        if (left_sz >= k) {
            split(nodes[t].left, k, l, nodes[t].left);
            r = t;
        } else {
            split(nodes[t].right, k - left_sz - 1, nodes[t].right, r);
            l = t;
        }
        pull(t);
    }

    // Merge l and r into t
    int merge(int l, int r) {
        if (!l || !r) return l | r;
        push(l); push(r);
        if (nodes[l].pri > nodes[r].pri) {
            nodes[l].right = merge(nodes[l].right, r);
            pull(l); return l;
        } else {
            nodes[r].left = merge(l, nodes[r].left);
            pull(r); return r;
        }
    }

    // Insert val at position pos (0-indexed)
    int insert(int root, int pos, ll val) {
        int l, r;
        split(root, pos, l, r);
        return merge(merge(l, new_node(val)), r);
    }

    // Erase element at position pos (0-indexed)
    int erase(int root, int pos) {
        int l, mid, r;
        split(root, pos, l, mid);
        split(mid, 1, mid, r);
        return merge(l, r);
    }

    // Reverse subarray [l, r] (0-indexed)
    int reverse_range(int root, int l, int r) {
        int a, b, c;
        split(root, l, a, b);
        split(b, r - l + 1, b, c);
        nodes[b].rev ^= 1;
        return merge(a, merge(b, c));
    }

    // Query sum in range [l, r] (0-indexed)
    ll query_range(int root, int l, int r) {
        int a, b, c;
        split(root, l, a, b);
        split(b, r - l + 1, b, c);
        ll res = sum(b);
        root = merge(a, merge(b, c));
        return res;
    }
};

// ═══════════════════════════════════════════════════════════
// 13. ORDERED SET (Policy-Based) — O(log N)
// ═══════════════════════════════════════════════════════════
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag,
                         tree_order_statistics_node_update>;
/*
 * Usage:
 *   ordered_set<int> os;
 *   os.insert(5);
 *   os.find_by_order(k);   // iterator to k-th element (0-indexed)
 *   os.order_of_key(x);    // count of elements strictly less than x
 *
 * For multiset behavior: use pair<int,int> with unique second element
 *   ordered_set<pair<int,int>> os;
 *   os.insert({val, unique_id});
 */

// ═══════════════════════════════════════════════════════════
// 14. MONOTONE STACK / MONOTONE DEQUE
// ═══════════════════════════════════════════════════════════

// Next Greater Element (right) — O(N)
vector<int> next_greater(const vector<int>& a) {
    int n = a.size();
    vector<int> res(n, -1);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.top()] < a[i]) {
            res[st.top()] = i;
            st.pop();
        }
        st.push(i);
    }
    return res;
}

// Previous Smaller Element (left) — O(N)
vector<int> prev_smaller(const vector<int>& a) {
    int n = a.size();
    vector<int> res(n, -1);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.top()] >= a[i]) st.pop();
        if (!st.empty()) res[i] = st.top();
        st.push(i);
    }
    return res;
}

// Sliding Window Max/Min — O(N) using Monotonic Deque
vector<int> sliding_window_max(const vector<int>& a, int k) {
    int n = a.size();
    vector<int> result;
    deque<int> dq;
    for (int i = 0; i < n; i++) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && a[dq.back()] <= a[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(a[dq.front()]);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
// 15. LINKED LIST (for CP — simple doubly linked list)
// ═══════════════════════════════════════════════════════════
struct LinkedList {
    vector<int> val, prv, nxt;
    int head, tail, sz;

    LinkedList() : head(-1), tail(-1), sz(0) {}

    int new_node(int v) {
        int id = val.size();
        val.push_back(v);
        prv.push_back(-1);
        nxt.push_back(-1);
        return id;
    }

    void push_back(int v) {
        int id = new_node(v);
        if (tail != -1) { nxt[tail] = id; prv[id] = tail; }
        else head = id;
        tail = id; sz++;
    }

    void push_front(int v) {
        int id = new_node(v);
        if (head != -1) { prv[head] = id; nxt[id] = head; }
        else tail = id;
        head = id; sz++;
    }

    void erase(int id) {
        if (prv[id] != -1) nxt[prv[id]] = nxt[id]; else head = nxt[id];
        if (nxt[id] != -1) prv[nxt[id]] = prv[id]; else tail = prv[id];
        sz--;
    }
};

// ═══════════════════════════════════════════════════════════
// 16. HEAP UTILITIES
// ═══════════════════════════════════════════════════════════
/*
 * STL priority_queue:
 *   priority_queue<int>                    pq;        // max-heap
 *   priority_queue<int, vector<int>, greater<int>> pq; // min-heap
 *
 * Custom comparator:
 *   auto cmp = [](const pii& a, const pii& b) { return a.second > b.second; };
 *   priority_queue<pii, vector<pii>, decltype(cmp)> pq(cmp);
 *
 * Median Maintenance with two heaps:
 */
struct MedianHeap {
    priority_queue<int> lo;                                  // max-heap for lower half
    priority_queue<int, vector<int>, greater<int>> hi;       // min-heap for upper half

    void insert(int x) {
        if (lo.empty() || x <= lo.top()) lo.push(x);
        else hi.push(x);
        // Balance
        while ((int)lo.size() > (int)hi.size() + 1) { hi.push(lo.top()); lo.pop(); }
        while ((int)hi.size() > (int)lo.size()) { lo.push(hi.top()); hi.pop(); }
    }

    int median() { return lo.top(); } // lower median
};

/*
 * ══════════════════════════════════════
 *  DS SELECTION GUIDE
 * ══════════════════════════════════════
 *
 * Point update + Range query   → Segment Tree / BIT
 * Range update + Range query   → Lazy Segment Tree / BIT (with 2 BITs)
 * Range query, no update       → Sparse Table (O(1) query) / Prefix Sums
 * Kth smallest in range        → Persistent SegTree / Merge Sort Tree
 * Dynamic insert/delete array  → Treap (Implicit)
 * Rank / Order statistics      → Ordered Set (pb_ds) / BIT
 * Prefix XOR queries           → Binary Trie
 * String matching prefix       → Trie
 * Path queries on tree         → HLD + Segment Tree
 * Subtree queries              → Euler Tour + Segment Tree / BIT
 * Distance queries on tree     → Centroid Decomposition
 * Sliding window min/max       → Monotonic Deque
 * Next greater / smaller       → Monotonic Stack
 */
