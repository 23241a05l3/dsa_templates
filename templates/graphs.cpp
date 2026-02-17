/*
 * ============================================================
 *              GRAPH ALGORITHMS — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Graph representation (adjacency list, weighted)
 *    2.  BFS & DFS
 *    3.  Dijkstra's Algorithm
 *    4.  Bellman-Ford
 *    5.  Floyd-Warshall
 *    6.  Topological Sort (Kahn's + DFS)
 *    7.  Kruskal's MST (with DSU)
 *    8.  Prim's MST
 *    9.  Strongly Connected Components (Kosaraju)
 *   10.  Bridges & Articulation Points
 *   11.  LCA (Binary Lifting)
 *   12.  Euler Tour & Subtree Queries
 *   13.  Bipartite Check
 *   14.  Cycle Detection (directed + undirected)
 *   15.  0-1 BFS
 *   16.  Multi-source BFS
 *   17.  Dinic's Max Flow
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;
typedef pair<ll,int> pli;
const ll INF = 1e18;
const int IINF = 1e9;

// ═══════════════════════════════════════════════════════════
// 1. GRAPH REPRESENTATION
// ═══════════════════════════════════════════════════════════
/*
   Unweighted:  vector<vector<int>> adj(n);
                adj[u].push_back(v);

   Weighted:    vector<vector<pair<int,ll>>> adj(n);
                adj[u].push_back({v, w});
*/

// ═══════════════════════════════════════════════════════════
// 2. BFS — O(V + E)
// ═══════════════════════════════════════════════════════════
vector<int> bfs(int src, const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> dist(n, -1);
    queue<int> q;
    dist[src] = 0;
    q.push(src);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

// ═══════════════════════════════════════════════════════════
// 3. DFS — O(V + E)
// ═══════════════════════════════════════════════════════════
// Iterative DFS (avoids stack overflow for large graphs)
vector<bool> dfs(int src, const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<bool> visited(n, false);
    stack<int> st;
    st.push(src);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (visited[u]) continue;
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) st.push(v);
        }
    }
    return visited;
}

// Recursive DFS with in/out times (useful for subtree queries)
int timer_dfs = 0;
vector<int> tin, tout;
void dfs_timer(int u, int p, const vector<vector<int>>& adj) {
    tin[u] = timer_dfs++;
    for (int v : adj[u]) {
        if (v != p) dfs_timer(v, u, adj);
    }
    tout[u] = timer_dfs++;
}
// Usage: tin.assign(n,0); tout.assign(n,0); timer_dfs=0; dfs_timer(root,-1,adj);
// is_ancestor(u,v): return tin[u] <= tin[v] && tout[v] <= tout[u];

// ═══════════════════════════════════════════════════════════
// 4. DIJKSTRA — O((V + E) log V)
// ═══════════════════════════════════════════════════════════
vector<ll> dijkstra(int src, const vector<vector<pair<int,ll>>>& adj) {
    int n = adj.size();
    vector<ll> dist(n, INF);
    priority_queue<pli, vector<pli>, greater<pli>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// ═══════════════════════════════════════════════════════════
// 5. BELLMAN-FORD — O(V * E)
// ═══════════════════════════════════════════════════════════
struct Edge { int u, v; ll w; };

// Returns dist array; dist[v] = -INF if part of negative cycle
vector<ll> bellman_ford(int src, int n, const vector<Edge>& edges) {
    vector<ll> dist(n, INF);
    dist[src] = 0;
    for (int i = 0; i < n - 1; i++) {
        for (auto& [u, v, w] : edges) {
            if (dist[u] < INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    // Detect negative cycles
    for (int i = 0; i < n; i++) {
        for (auto& [u, v, w] : edges) {
            if (dist[u] < INF && dist[u] + w < dist[v]) {
                dist[v] = -INF;
            }
        }
    }
    return dist;
}

// ═══════════════════════════════════════════════════════════
// 6. FLOYD-WARSHALL — O(V³)
// ═══════════════════════════════════════════════════════════
// dist[i][j] should be initialized: 0 for i==j, weight for edges, INF otherwise
void floyd_warshall(vector<vector<ll>>& dist) {
    int n = dist.size();
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] < INF && dist[k][j] < INF)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
}

// ═══════════════════════════════════════════════════════════
// 7. TOPOLOGICAL SORT — O(V + E)
// ═══════════════════════════════════════════════════════════
// Kahn's BFS-based — also detects cycle (returns empty if cycle exists)
vector<int> topo_sort_kahn(int n, const vector<vector<int>>& adj) {
    vector<int> indeg(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : adj[u]) indeg[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indeg[i] == 0) q.push(i);
    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return (int)order.size() == n ? order : vector<int>(); // empty = cycle
}

// DFS-based Topological Sort
vector<int> topo_sort_dfs(int n, const vector<vector<int>>& adj) {
    vector<int> vis(n, 0), order;
    bool has_cycle = false;
    function<void(int)> dfs = [&](int u) {
        vis[u] = 1; // 1 = in progress
        for (int v : adj[u]) {
            if (vis[v] == 1) { has_cycle = true; return; }
            if (vis[v] == 0) dfs(v);
        }
        vis[u] = 2; // 2 = done
        order.push_back(u);
    };
    for (int i = 0; i < n; i++)
        if (vis[i] == 0) dfs(i);
    if (has_cycle) return {};
    reverse(order.begin(), order.end());
    return order;
}

// ═══════════════════════════════════════════════════════════
// 8. KRUSKAL'S MST — O(E log E)
// ═══════════════════════════════════════════════════════════
struct DSU {
    vector<int> par, rnk;
    DSU(int n) : par(n), rnk(n, 0) { iota(par.begin(), par.end(), 0); }
    int find(int x) { return par[x] == x ? x : par[x] = find(par[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rnk[a] < rnk[b]) swap(a, b);
        par[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
        return true;
    }
    bool connected(int a, int b) { return find(a) == find(b); }
};

ll kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b){
        return a.w < b.w;
    });
    DSU dsu(n);
    ll mst_cost = 0;
    int edges_added = 0;
    for (auto& [u, v, w] : edges) {
        if (dsu.unite(u, v)) {
            mst_cost += w;
            if (++edges_added == n - 1) break;
        }
    }
    return mst_cost; // returns -1 if forest: edges_added < n-1
}

// ═══════════════════════════════════════════════════════════
// 9. PRIM'S MST — O((V + E) log V)
// ═══════════════════════════════════════════════════════════
ll prim(int n, const vector<vector<pair<int,ll>>>& adj) {
    vector<bool> inMST(n, false);
    priority_queue<pli, vector<pli>, greater<pli>> pq;
    pq.push({0, 0});
    ll mst_cost = 0;
    int cnt = 0;
    while (!pq.empty() && cnt < n) {
        auto [w, u] = pq.top(); pq.pop();
        if (inMST[u]) continue;
        inMST[u] = true;
        mst_cost += w;
        cnt++;
        for (auto [v, wt] : adj[u])
            if (!inMST[v]) pq.push({wt, v});
    }
    return mst_cost;
}

// ═══════════════════════════════════════════════════════════
// 10. KOSARAJU'S SCC — O(V + E)
// ═══════════════════════════════════════════════════════════
struct SCC {
    int n, num_scc;
    vector<vector<int>> adj, radj;
    vector<int> order, comp;
    vector<bool> vis;

    SCC(int n) : n(n), adj(n), radj(n), comp(n, -1), vis(n, false), num_scc(0) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        radj[v].push_back(u);
    }

    void dfs1(int u) {
        vis[u] = true;
        for (int v : adj[u]) if (!vis[v]) dfs1(v);
        order.push_back(u);
    }

    void dfs2(int u, int c) {
        comp[u] = c;
        for (int v : radj[u]) if (comp[v] == -1) dfs2(v, c);
    }

    int solve() {
        for (int i = 0; i < n; i++) if (!vis[i]) dfs1(i);
        reverse(order.begin(), order.end());
        for (int u : order) if (comp[u] == -1) dfs2(u, num_scc++);
        return num_scc;
    }
};

// ═══════════════════════════════════════════════════════════
// 11. BRIDGES & ARTICULATION POINTS — O(V + E)
// ═══════════════════════════════════════════════════════════
struct BridgesAP {
    int n, timer = 0;
    vector<vector<int>> adj;
    vector<int> disc, low;
    vector<bool> visited, is_ap;
    vector<pii> bridges;

    BridgesAP(int n) : n(n), adj(n), disc(n), low(n), visited(n, false), is_ap(n, false) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs(int u, int parent) {
        visited[u] = true;
        disc[u] = low[u] = timer++;
        int children = 0;
        for (int v : adj[u]) {
            if (!visited[v]) {
                children++;
                dfs(v, u);
                low[u] = min(low[u], low[v]);
                // Bridge
                if (low[v] > disc[u])
                    bridges.push_back({u, v});
                // Articulation Point
                if (parent == -1 && children > 1) is_ap[u] = true;
                if (parent != -1 && low[v] >= disc[u]) is_ap[u] = true;
            } else if (v != parent) {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    void solve() {
        for (int i = 0; i < n; i++)
            if (!visited[i]) dfs(i, -1);
    }
};

// ═══════════════════════════════════════════════════════════
// 12. LCA (Binary Lifting) — O(N log N) preprocessing, O(log N) query
// ═══════════════════════════════════════════════════════════
struct LCA {
    int n, LOG;
    vector<vector<int>> adj, up;
    vector<int> depth;

    LCA(int n) : n(n), adj(n), depth(n, 0) {
        LOG = __lg(n) + 2;
        up.assign(n, vector<int>(LOG, 0));
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void build(int root = 0) {
        function<void(int, int)> dfs = [&](int u, int p) {
            up[u][0] = p;
            for (int k = 1; k < LOG; k++)
                up[u][k] = up[up[u][k-1]][k-1];
            for (int v : adj[u]) {
                if (v != p) {
                    depth[v] = depth[u] + 1;
                    dfs(v, u);
                }
            }
        };
        dfs(root, root);
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);
        int diff = depth[u] - depth[v];
        for (int k = 0; k < LOG; k++)
            if ((diff >> k) & 1) u = up[u][k];
        if (u == v) return u;
        for (int k = LOG - 1; k >= 0; k--)
            if (up[u][k] != up[v][k]) { u = up[u][k]; v = up[v][k]; }
        return up[u][0];
    }

    int dist(int u, int v) {
        return depth[u] + depth[v] - 2 * depth[lca(u, v)];
    }
};

// ═══════════════════════════════════════════════════════════
// 13. BIPARTITE CHECK — O(V + E)
// ═══════════════════════════════════════════════════════════
// Returns color array (0/1). Returns empty if not bipartite.
vector<int> bipartite_check(int n, const vector<vector<int>>& adj) {
    vector<int> color(n, -1);
    bool is_bip = true;
    for (int i = 0; i < n && is_bip; i++) {
        if (color[i] != -1) continue;
        queue<int> q;
        q.push(i); color[i] = 0;
        while (!q.empty() && is_bip) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (color[v] == -1) {
                    color[v] = color[u] ^ 1;
                    q.push(v);
                } else if (color[v] == color[u]) {
                    is_bip = false;
                }
            }
        }
    }
    return is_bip ? color : vector<int>();
}

// ═══════════════════════════════════════════════════════════
// 14. CYCLE DETECTION
// ═══════════════════════════════════════════════════════════
// Directed graph — returns true if cycle exists
bool has_cycle_directed(int n, const vector<vector<int>>& adj) {
    vector<int> vis(n, 0);
    bool cycle = false;
    function<void(int)> dfs = [&](int u) {
        vis[u] = 1;
        for (int v : adj[u]) {
            if (vis[v] == 1) { cycle = true; return; }
            if (vis[v] == 0) dfs(v);
        }
        vis[u] = 2;
    };
    for (int i = 0; i < n && !cycle; i++)
        if (vis[i] == 0) dfs(i);
    return cycle;
}

// Undirected graph — returns true if cycle exists
bool has_cycle_undirected(int n, const vector<vector<int>>& adj) {
    DSU dsu(n);
    for (int u = 0; u < n; u++)
        for (int v : adj[u])
            if (u < v && !dsu.unite(u, v)) return true;
    return false;
}

// ═══════════════════════════════════════════════════════════
// 15. 0-1 BFS — O(V + E) for graphs with edge weights 0 or 1
// ═══════════════════════════════════════════════════════════
vector<int> bfs_01(int src, const vector<vector<pii>>& adj) {
    int n = adj.size();
    vector<int> dist(n, IINF);
    deque<int> dq;
    dist[src] = 0;
    dq.push_back(src);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (w == 0) dq.push_front(v);
                else dq.push_back(v);
            }
        }
    }
    return dist;
}

// ═══════════════════════════════════════════════════════════
// 16. MULTI-SOURCE BFS
// ═══════════════════════════════════════════════════════════
vector<int> multi_source_bfs(const vector<int>& sources, const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> dist(n, -1);
    queue<int> q;
    for (int s : sources) {
        dist[s] = 0;
        q.push(s);
    }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

// ═══════════════════════════════════════════════════════════
// 17. DINIC'S MAX FLOW — O(V² * E)
// ═══════════════════════════════════════════════════════════
struct Dinic {
    struct FlowEdge {
        int to, rev;
        ll cap;
    };
    int n;
    vector<vector<FlowEdge>> graph;
    vector<int> level, iter;

    Dinic(int n) : n(n), graph(n), level(n), iter(n) {}

    void add_edge(int from, int to, ll cap) {
        graph[from].push_back({to, (int)graph[to].size(), cap});
        graph[to].push_back({from, (int)graph[from].size() - 1, 0});
    }

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (auto& e : graph[v]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[t] >= 0;
    }

    ll dfs(int v, int t, ll f) {
        if (v == t) return f;
        for (int& i = iter[v]; i < (int)graph[v].size(); i++) {
            FlowEdge& e = graph[v][i];
            if (e.cap > 0 && level[v] < level[e.to]) {
                ll d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;
                    graph[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }

    ll max_flow(int s, int t) {
        ll flow = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            ll d;
            while ((d = dfs(s, t, INF)) > 0) flow += d;
        }
        return flow;
    }
};

/*
 * ══════════════════════════════════════════
 *  QUICK REFERENCE — WHEN TO USE WHAT
 * ══════════════════════════════════════════
 *
 * Shortest Path:
 *   - Unweighted           → BFS
 *   - Weights 0/1          → 0-1 BFS
 *   - Non-negative weights → Dijkstra
 *   - Negative weights     → Bellman-Ford
 *   - All pairs            → Floyd-Warshall
 *
 * MST:
 *   - Sparse graph         → Kruskal
 *   - Dense graph          → Prim
 *
 * Connectivity:
 *   - Connected components → BFS / DFS / DSU
 *   - Strongly connected   → Kosaraju / Tarjan
 *   - Bridges / APs        → Tarjan-style lowlink
 *   - Bipartiteness        → BFS 2-coloring
 *
 * Trees:
 *   - LCA queries          → Binary Lifting
 *   - Subtree queries      → Euler Tour + Segment Tree
 *   - Path queries         → HLD / LCA-based
 *
 * Flow:
 *   - Max flow             → Dinic's
 *   - Min cut = Max flow   → Dinic's
 *   - Bipartite matching   → Dinic on bipartite / Hopcroft-Karp
 */
