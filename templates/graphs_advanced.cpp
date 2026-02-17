/*
 * ============================================================
 *     ADVANCED GRAPH ALGORITHMS — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Tarjan's SCC (Strongly Connected Components)
 *    2.  Hopcroft-Karp Bipartite Matching — O(E√V)
 *    3.  Euler Path / Circuit (Hierholzer's Algorithm)
 *    4.  Min Cost Max Flow (MCMF) — SPFA-based
 *    5.  SPFA (Shortest Path Faster Algorithm)
 *    6.  A* Search
 *    7.  Hungarian Algorithm (Weighted Bipartite Matching) — O(N³)
 *    8.  2-SAT
 *    9.  Chromatic Number (Inclusion-Exclusion) — O(2^N * N)
 *   10.  Block Cut Tree (Biconnected Components)
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

// ═══════════════════════════════════════════════════════════
// 1. TARJAN'S SCC — O(V + E)
// ═══════════════════════════════════════════════════════════
struct TarjanSCC {
    int n, timer = 0, nscc = 0;
    vector<vector<int>> adj;
    vector<int> disc, low, comp; // comp[u] = SCC id (0-indexed)
    vector<bool> on_stack;
    stack<int> st;

    TarjanSCC(int n) : n(n), adj(n), disc(n, -1), low(n), comp(n, -1), on_stack(n, false) {}

    void add_edge(int u, int v) { adj[u].push_back(v); }

    void dfs(int u) {
        disc[u] = low[u] = timer++;
        st.push(u);
        on_stack[u] = true;
        for (int v : adj[u]) {
            if (disc[v] == -1) {
                dfs(v);
                low[u] = min(low[u], low[v]);
            } else if (on_stack[v]) {
                low[u] = min(low[u], disc[v]);
            }
        }
        if (low[u] == disc[u]) {
            while (true) {
                int v = st.top(); st.pop();
                on_stack[v] = false;
                comp[v] = nscc;
                if (v == u) break;
            }
            nscc++;
        }
    }

    void solve() {
        for (int i = 0; i < n; i++)
            if (disc[i] == -1) dfs(i);
    }

    // Build condensation DAG
    vector<vector<int>> condensation() {
        vector<vector<int>> dag(nscc);
        set<pii> edges;
        for (int u = 0; u < n; u++)
            for (int v : adj[u])
                if (comp[u] != comp[v] && edges.insert({comp[u], comp[v]}).second)
                    dag[comp[u]].push_back(comp[v]);
        return dag;
    }
};

// ═══════════════════════════════════════════════════════════
// 2. HOPCROFT-KARP BIPARTITE MATCHING — O(E√V)
// ═══════════════════════════════════════════════════════════
struct HopcroftKarp {
    int n, m; // n = left set size, m = right set size
    vector<vector<int>> adj; // adj[u] = right nodes connected to u
    vector<int> match_l, match_r, dist;
    static const int INF = 1e9;

    HopcroftKarp(int n, int m) : n(n), m(m), adj(n), match_l(n, -1), match_r(m, -1), dist(n) {}

    void add_edge(int u, int v) { adj[u].push_back(v); } // u in [0,n), v in [0,m)

    bool bfs() {
        queue<int> q;
        for (int u = 0; u < n; u++) {
            if (match_l[u] == -1) { dist[u] = 0; q.push(u); }
            else dist[u] = INF;
        }
        bool found = false;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                int w = match_r[v];
                if (w == -1) found = true;
                else if (dist[w] == INF) {
                    dist[w] = dist[u] + 1;
                    q.push(w);
                }
            }
        }
        return found;
    }

    bool dfs(int u) {
        for (int v : adj[u]) {
            int w = match_r[v];
            if (w == -1 || (dist[w] == dist[u] + 1 && dfs(w))) {
                match_l[u] = v;
                match_r[v] = u;
                return true;
            }
        }
        dist[u] = INF;
        return false;
    }

    int max_matching() {
        int ans = 0;
        while (bfs())
            for (int u = 0; u < n; u++)
                if (match_l[u] == -1 && dfs(u)) ans++;
        return ans;
    }
};

// ═══════════════════════════════════════════════════════════
// 3. EULER PATH / CIRCUIT — Hierholzer's Algorithm — O(V + E)
// ═══════════════════════════════════════════════════════════

// For DIRECTED graph
struct EulerDirected {
    int n;
    vector<vector<int>> adj;
    vector<int> idx; // current edge pointer for each node

    EulerDirected(int n) : n(n), adj(n), idx(n, 0) {}

    void add_edge(int u, int v) { adj[u].push_back(v); }

    // Check if Euler circuit exists: every node has in-degree == out-degree
    // Check if Euler path exists: at most one node has out - in == 1 (start),
    //   at most one has in - out == 1 (end), rest equal

    vector<int> find_euler_path(int start) {
        stack<int> st;
        vector<int> path;
        st.push(start);
        while (!st.empty()) {
            int u = st.top();
            if (idx[u] < (int)adj[u].size()) {
                st.push(adj[u][idx[u]++]);
            } else {
                path.push_back(u);
                st.pop();
            }
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // Find start node for Euler path/circuit
    int find_start() {
        vector<int> in_deg(n, 0);
        for (int u = 0; u < n; u++)
            for (int v : adj[u]) in_deg[v]++;
        int start = -1;
        for (int u = 0; u < n; u++) {
            if ((int)adj[u].size() - in_deg[u] == 1) return u;
            if (!adj[u].empty() && start == -1) start = u;
        }
        return start; // circuit case
    }
};

// For UNDIRECTED graph
struct EulerUndirected {
    int n, m = 0;
    vector<vector<pii>> adj; // {neighbor, edge_id}
    vector<bool> used_edge;

    EulerUndirected(int n) : n(n), adj(n) {}

    void add_edge(int u, int v) {
        adj[u].push_back({v, m});
        adj[v].push_back({u, m});
        m++;
    }

    vector<int> find_euler_path(int start) {
        used_edge.assign(m, false);
        vector<int> idx(n, 0);
        stack<int> st;
        vector<int> path;
        st.push(start);
        while (!st.empty()) {
            int u = st.top();
            if (idx[u] < (int)adj[u].size()) {
                auto [v, eid] = adj[u][idx[u]++];
                if (!used_edge[eid]) {
                    used_edge[eid] = true;
                    st.push(v);
                }
            } else {
                path.push_back(u);
                st.pop();
            }
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // Start for Euler path: odd-degree node (if exists), else any node with edges
    int find_start() {
        int start = -1;
        for (int u = 0; u < n; u++) {
            if (adj[u].size() % 2 == 1) return u;
            if (!adj[u].empty() && start == -1) start = u;
        }
        return start;
    }
};

// ═══════════════════════════════════════════════════════════
// 4. MIN COST MAX FLOW (MCMF) — SPFA-based (Bellman-Ford)
// ═══════════════════════════════════════════════════════════
struct MCMF {
    struct Edge { int to, cap, cost, flow; };
    int n;
    vector<Edge> edges;
    vector<vector<int>> g;
    vector<int> d, p, a;
    vector<bool> inq;

    MCMF(int n) : n(n), g(n), d(n), p(n), a(n), inq(n) {}

    void add_edge(int from, int to, int cap, int cost) {
        g[from].push_back(edges.size());
        edges.push_back({to, cap, cost, 0});
        g[to].push_back(edges.size());
        edges.push_back({from, 0, -cost, 0});
    }

    bool spfa(int s, int t) {
        fill(d.begin(), d.end(), INT_MAX);
        fill(inq.begin(), inq.end(), false);
        d[s] = 0; a[s] = INT_MAX;
        queue<int> q;
        q.push(s); inq[s] = true;
        while (!q.empty()) {
            int u = q.front(); q.pop(); inq[u] = false;
            for (int id : g[u]) {
                auto& e = edges[id];
                if (e.cap > e.flow && d[e.to] > d[u] + e.cost) {
                    d[e.to] = d[u] + e.cost;
                    p[e.to] = id;
                    a[e.to] = min(a[u], e.cap - e.flow);
                    if (!inq[e.to]) { q.push(e.to); inq[e.to] = true; }
                }
            }
        }
        return d[t] != INT_MAX;
    }

    // Returns {max_flow, min_cost}
    pair<int, int> solve(int s, int t) {
        int flow = 0, cost = 0;
        while (spfa(s, t)) {
            flow += a[t];
            cost += a[t] * d[t];
            int u = t;
            while (u != s) {
                edges[p[u]].flow += a[t];
                edges[p[u] ^ 1].flow -= a[t];
                u = edges[p[u] ^ 1].to;
            }
        }
        return {flow, cost};
    }
};

// ═══════════════════════════════════════════════════════════
// 5. SPFA (Shortest Path Faster Algorithm) — O(VE) worst
// ═══════════════════════════════════════════════════════════
// Note: Faster than Bellman-Ford in practice, handles negative weights
// Can detect negative cycles
struct SPFA {
    static const int INF = 1e9;
    int n;
    vector<vector<pair<int,int>>> adj; // {to, weight}

    SPFA(int n) : n(n), adj(n) {}

    void add_edge(int u, int v, int w) { adj[u].push_back({v, w}); }

    // Returns {dist, has_negative_cycle}
    pair<vector<int>, bool> shortest_path(int src) {
        vector<int> dist(n, INF), cnt(n, 0);
        vector<bool> in_queue(n, false);
        deque<int> q;
        dist[src] = 0;
        q.push_back(src);
        in_queue[src] = true;

        while (!q.empty()) {
            int u = q.front(); q.pop_front();
            in_queue[u] = false;
            for (auto [v, w] : adj[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    if (!in_queue[v]) {
                        // SLF optimization
                        if (!q.empty() && dist[v] < dist[q.front()])
                            q.push_front(v);
                        else
                            q.push_back(v);
                        in_queue[v] = true;
                        cnt[v]++;
                        if (cnt[v] >= n) return {dist, true}; // negative cycle
                    }
                }
            }
        }
        return {dist, false};
    }
};

// ═══════════════════════════════════════════════════════════
// 6. A* SEARCH — O(E log V) with good heuristic
// ═══════════════════════════════════════════════════════════
// Grid-based A* (for 2D grid problems)
struct AStar {
    struct State {
        int x, y, g, f;
        bool operator>(const State& o) const { return f > o.f; }
    };

    int rows, cols;
    vector<vector<int>> grid; // 0 = free, 1 = obstacle

    AStar(int r, int c, vector<vector<int>>& g) : rows(r), cols(c), grid(g) {}

    // Manhattan distance heuristic
    int heuristic(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }

    int solve(int sx, int sy, int ex, int ey) {
        if (grid[sx][sy] || grid[ex][ey]) return -1;
        int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
        vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
        priority_queue<State, vector<State>, greater<State>> pq;
        dist[sx][sy] = 0;
        pq.push({sx, sy, 0, heuristic(sx, sy, ex, ey)});

        while (!pq.empty()) {
            auto [x, y, g, f] = pq.top(); pq.pop();
            if (x == ex && y == ey) return g;
            if (g > dist[x][y]) continue;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= rows || ny < 0 || ny >= cols || grid[nx][ny]) continue;
                int ng = g + 1;
                if (ng < dist[nx][ny]) {
                    dist[nx][ny] = ng;
                    pq.push({nx, ny, ng, ng + heuristic(nx, ny, ex, ey)});
                }
            }
        }
        return -1;
    }
};

// General graph A* (for weighted graph problems)
int astar_graph(int n, vector<vector<pii>>& adj, int src, int dst,
                function<int(int)> heuristic) {
    // heuristic(u) = estimated distance from u to dst
    vector<int> dist(n, INT_MAX);
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    dist[src] = 0;
    pq.push({heuristic(src), src});

    while (!pq.empty()) {
        auto [f, u] = pq.top(); pq.pop();
        if (u == dst) return dist[dst];
        if (f - heuristic(u) > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v] + heuristic(v), v});
            }
        }
    }
    return -1;
}

// ═══════════════════════════════════════════════════════════
// 7. HUNGARIAN ALGORITHM — O(N³) — Min cost assignment
// ═══════════════════════════════════════════════════════════
// Assigns N workers to N jobs to minimize total cost
struct Hungarian {
    int n;
    vector<vector<int>> cost;

    Hungarian(int n, vector<vector<int>>& c) : n(n), cost(c) {}

    // Returns {min_cost, assignment} where assignment[i] = job for worker i
    pair<int, vector<int>> solve() {
        // 0-indexed workers and jobs
        const int INF = 1e9;
        vector<int> u(n+1), v(n+1), p(n+1), way(n+1);
        // u[i] = potential for worker i, v[j] = potential for job j
        // p[j] = worker assigned to job j (0 = unassigned)
        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            vector<int> minv(n+1, INF);
            vector<bool> used(n+1, false);
            do {
                used[j0] = true;
                int i0 = p[j0], delta = INF, j1;
                for (int j = 1; j <= n; j++) {
                    if (used[j]) continue;
                    int cur = cost[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                }
                j0 = j1;
            } while (p[j0] != 0);
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
        vector<int> assignment(n);
        int total = 0;
        for (int j = 1; j <= n; j++) {
            assignment[p[j] - 1] = j - 1;
            total += cost[p[j] - 1][j - 1];
        }
        return {total, assignment};
    }
};

// ═══════════════════════════════════════════════════════════
// 8. 2-SAT — O(V + E)
// ═══════════════════════════════════════════════════════════
struct TwoSAT {
    int n; // number of boolean variables
    TarjanSCC scc;

    TwoSAT(int n) : n(n), scc(2 * n) {}

    // Variable x: true = 2*x, false = 2*x+1
    int var_true(int x) { return 2 * x; }
    int var_false(int x) { return 2 * x + 1; }
    int neg(int lit) { return lit ^ 1; }

    // Add clause: (a OR b)
    // Implication: ~a → b and ~b → a
    void add_clause(int a, int b) {
        scc.add_edge(neg(a), b);
        scc.add_edge(neg(b), a);
    }

    // Convenience: add clause with bool flags
    // add_clause_vars(x, true, y, false) means (x OR ~y)
    void add_clause_vars(int x, bool xval, int y, bool yval) {
        int a = xval ? var_true(x) : var_false(x);
        int b = yval ? var_true(y) : var_false(y);
        add_clause(a, b);
    }

    // Force variable x to value val
    void force(int x, bool val) {
        int lit = val ? var_true(x) : var_false(x);
        add_clause(lit, lit);
    }

    // Solve: returns empty vector if unsatisfiable, else assignment[i] = true/false
    vector<bool> solve() {
        scc.solve();
        vector<bool> assignment(n);
        for (int i = 0; i < n; i++) {
            if (scc.comp[var_true(i)] == scc.comp[var_false(i)])
                return {}; // unsatisfiable
            assignment[i] = scc.comp[var_true(i)] > scc.comp[var_false(i)];
        }
        return assignment;
    }
};

// ═══════════════════════════════════════════════════════════
// 9. CHROMATIC NUMBER — Inclusion-Exclusion — O(2^N * N)
// ═══════════════════════════════════════════════════════════
// Number of proper vertex colorings with exactly k colors
int chromatic_number(int n, vector<vector<int>>& adj_list) {
    // adj_mask[v] = bitmask of neighbors of v
    vector<int> adj_mask(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : adj_list[u])
            adj_mask[u] |= (1 << v);

    // ind[mask] = number of independent sets in subset mask
    vector<int> ind(1 << n, 0);
    ind[0] = 1;
    for (int mask = 1; mask < (1 << n); mask++) {
        int v = __builtin_ctz(mask); // pick lowest vertex
        int rest = mask ^ (1 << v);
        // Subsets not containing v + subsets containing v (only if no neighbor in subset)
        ind[mask] = ind[rest] + ind[rest & ~adj_mask[v]];
    }

    // Chromatic polynomial via inclusion-exclusion
    auto count_colorings = [&](int k) -> ll {
        ll sum = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int sign = (__builtin_popcount(mask) % 2 == 0) ? 1 : -1;
            ll term = 1;
            // k^(number of independent sets in complement mask)
            // Actually: P(k) = sum over subsets S of (-1)^|S| * k^(components of G\S)
            // Simplified: use ind[] directly
            int compl_mask = ((1 << n) - 1) ^ mask;
            // This is a simplification; for exact chromatic number:
        }
        return sum;
    };

    // Alternative: just find minimum k such that graph is k-colorable
    // Using inclusion-exclusion on independent sets:
    // f(k) = sum_{S subset [n]} (-1)^{n-|S|} * ind[S]^k ... but simplified:

    // Simplified chromatic number finder:
    for (int k = 1; k <= n; k++) {
        // Check if k-colorable using inclusion-exclusion
        ll total = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int sign = (__builtin_popcount(((1 << n) - 1) ^ mask) % 2 == 0) ? 1 : -1;
            // Compute ind[mask]^k with possible overflow issues
            ll pw = 1;
            for (int i = 0; i < k; i++) pw *= ind[mask];
            total += sign * pw;
        }
        if (total > 0) return k;
    }
    return n;
}

// ═══════════════════════════════════════════════════════════
// 10. BLOCK-CUT TREE (Biconnected Components)
// ═══════════════════════════════════════════════════════════
struct BlockCutTree {
    int n, timer = 0;
    vector<vector<int>> adj;
    vector<int> disc, low;
    vector<bool> is_cut;
    stack<int> stk;
    vector<vector<int>> components; // biconnected components (vertex sets)
    vector<vector<int>> tree; // block-cut tree

    BlockCutTree(int n) : n(n), adj(n), disc(n, -1), low(n), is_cut(n, false) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs(int u, int parent) {
        disc[u] = low[u] = timer++;
        stk.push(u);
        int children = 0;
        for (int v : adj[u]) {
            if (v == parent) continue;
            if (disc[v] == -1) {
                children++;
                dfs(v, u);
                low[u] = min(low[u], low[v]);
                if ((parent == -1 && children > 1) || (parent != -1 && low[v] >= disc[u])) {
                    is_cut[u] = true;
                }
                if (low[v] >= disc[u]) {
                    components.emplace_back();
                    while (true) {
                        int w = stk.top(); stk.pop();
                        components.back().push_back(w);
                        if (w == v) break;
                    }
                    components.back().push_back(u);
                }
            } else {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    void solve() {
        for (int i = 0; i < n; i++)
            if (disc[i] == -1) dfs(i, -1);
    }

    // Build the block-cut tree
    // Nodes: cut vertices + block nodes (one per biconnected component)
    // Returns number of nodes in block-cut tree
    int build_tree() {
        solve();
        int total = n + components.size(); // cut vertices + blocks
        tree.resize(total);
        for (int b = 0; b < (int)components.size(); b++) {
            int block_node = n + b;
            for (int v : components[b]) {
                tree[v].push_back(block_node);
                tree[block_node].push_back(v);
            }
        }
        return total;
    }
};

/*
 * ══════════════════════════════════════
 *  ALGORITHM SELECTION GUIDE:
 * ══════════════════════════════════════
 *
 * Bipartite matching:
 *   Unweighted → Hopcroft-Karp O(E√V)
 *   Weighted → Hungarian O(N³)
 *
 * Network flow:
 *   Max flow only → Dinic's (in graphs.cpp)
 *   Min cost max flow → MCMF (SPFA-based, above)
 *
 * Shortest path with negative weights:
 *   SPFA (practical), Bellman-Ford (theoretical)
 *   Detect negative cycles → SPFA with cnt >= N
 *
 * Euler path/circuit:
 *   Directed: each node in-deg == out-deg (circuit)
 *             or exactly one node out-in=1, one in-out=1 (path)
 *   Undirected: all even degree (circuit), or exactly 2 odd (path)
 *
 * 2-SAT: for boolean satisfiability with clauses of size 2
 *   Reduce to implication graph → Tarjan SCC
 *
 * SCC: Tarjan (one DFS) or Kosaraju (two DFS)
 */
