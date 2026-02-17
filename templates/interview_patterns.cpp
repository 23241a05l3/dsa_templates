/*
 * ============================================================
 *     INTERVIEW PATTERNS — FAANG TEMPLATE
 *     (Intervals, Matrix, BFS/DFS, Prefix Sum, Sampling,
 *      Linked List, Graph, Trie, Binary Search patterns)
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
//             I N T E R V A L   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Merge Intervals — O(N log N)
vector<vector<int>> merge_intervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> res;
    for (auto& iv : intervals) {
        if (!res.empty() && res.back()[1] >= iv[0])
            res.back()[1] = max(res.back()[1], iv[1]);
        else
            res.push_back(iv);
    }
    return res;
}

// Insert Interval — O(N)
vector<vector<int>> insert_interval(vector<vector<int>>& intervals, vector<int>& newIv) {
    vector<vector<int>> res;
    int i = 0, n = intervals.size();
    while (i < n && intervals[i][1] < newIv[0]) res.push_back(intervals[i++]);
    while (i < n && intervals[i][0] <= newIv[1]) {
        newIv[0] = min(newIv[0], intervals[i][0]);
        newIv[1] = max(newIv[1], intervals[i][1]);
        i++;
    }
    res.push_back(newIv);
    while (i < n) res.push_back(intervals[i++]);
    return res;
}

// Meeting Rooms II — Min rooms needed — O(N log N) sweep line
int min_meeting_rooms(vector<vector<int>>& intervals) {
    vector<pair<int,int>> events;
    for (auto& iv : intervals) {
        events.push_back({iv[0], 1});  // start
        events.push_back({iv[1], -1}); // end
    }
    sort(events.begin(), events.end());
    int cur = 0, mx = 0;
    for (auto& [time, delta] : events) { cur += delta; mx = max(mx, cur); }
    return mx;
}

// Interval List Intersections — two sorted interval lists
vector<vector<int>> interval_intersection(vector<vector<int>>& A, vector<vector<int>>& B) {
    vector<vector<int>> res;
    int i = 0, j = 0;
    while (i < (int)A.size() && j < (int)B.size()) {
        int lo = max(A[i][0], B[j][0]), hi = min(A[i][1], B[j][1]);
        if (lo <= hi) res.push_back({lo, hi});
        if (A[i][1] < B[j][1]) i++;
        else j++;
    }
    return res;
}

// ═══════════════════════════════════════════════════════════
//          M A T R I X   T R A V E R S A L S
// ═══════════════════════════════════════════════════════════

// Spiral Order — O(M*N)
vector<int> spiral_order(const vector<vector<int>>& matrix) {
    vector<int> res;
    int top = 0, bottom = matrix.size() - 1, left = 0, right = matrix[0].size() - 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; j++) res.push_back(matrix[top][j]);
        top++;
        for (int i = top; i <= bottom; i++) res.push_back(matrix[i][right]);
        right--;
        if (top <= bottom) { for (int j = right; j >= left; j--) res.push_back(matrix[bottom][j]); bottom--; }
        if (left <= right) { for (int i = bottom; i >= top; i--) res.push_back(matrix[i][left]); left++; }
    }
    return res;
}

// Rotate Image 90° Clockwise — in-place
void rotate_image(vector<vector<int>>& matrix) {
    int n = matrix.size();
    // Transpose
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            swap(matrix[i][j], matrix[j][i]);
    // Reverse each row
    for (auto& row : matrix) reverse(row.begin(), row.end());
}

// Set Matrix Zeros — O(1) extra space
void set_zeroes(vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    bool first_row = false, first_col = false;
    for (int j = 0; j < n; j++) if (matrix[0][j] == 0) first_row = true;
    for (int i = 0; i < m; i++) if (matrix[i][0] == 0) first_col = true;
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            if (matrix[i][j] == 0) { matrix[i][0] = 0; matrix[0][j] = 0; }
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
    if (first_row) for (int j = 0; j < n; j++) matrix[0][j] = 0;
    if (first_col) for (int i = 0; i < m; i++) matrix[i][0] = 0;
}

// Search in 2D Sorted Matrix (each row sorted, first el > last of prev row)
bool search_matrix(const vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    int lo = 0, hi = m * n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int val = matrix[mid / n][mid % n];
        if (val == target) return true;
        if (val < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

// Staircase search (rows sorted left-right, cols sorted top-bottom)
bool search_matrix_2(const vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    int r = 0, c = n - 1;
    while (r < m && c >= 0) {
        if (matrix[r][c] == target) return true;
        if (matrix[r][c] > target) c--;
        else r++;
    }
    return false;
}

// Diagonal Traversal
vector<int> diagonal_traverse(const vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    vector<int> res;
    for (int d = 0; d < m + n - 1; d++) {
        vector<int> diag;
        int r = d < n ? 0 : d - n + 1;
        int c = d < n ? d : n - 1;
        while (r < m && c >= 0) { diag.push_back(mat[r][c]); r++; c--; }
        if (d % 2 == 0) reverse(diag.begin(), diag.end());
        for (int x : diag) res.push_back(x);
    }
    return res;
}

// Game of Life — in-place with bit encoding
void game_of_life(vector<vector<int>>& board) {
    int m = board.size(), n = board[0].size();
    int dx[] = {-1,-1,-1,0,0,1,1,1}, dy[] = {-1,0,1,-1,1,-1,0,1};
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int live = 0;
            for (int d = 0; d < 8; d++) {
                int ni = i + dx[d], nj = j + dy[d];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n)
                    live += board[ni][nj] & 1; // check current state (bit 0)
            }
            // Encode next state in bit 1
            if (board[i][j] & 1) { if (live == 2 || live == 3) board[i][j] |= 2; }
            else { if (live == 3) board[i][j] |= 2; }
        }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            board[i][j] >>= 1;
}

// ═══════════════════════════════════════════════════════════
//      B F S / D F S   I N T E R V I E W   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Number of Islands — flood fill
int num_islands(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size(), count = 0;
    function<void(int, int)> dfs = [&](int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
        grid[i][j] = '0';
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1);
    };
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == '1') { count++; dfs(i, j); }
    return count;
}

// Rotten Oranges — Multi-source BFS
int oranges_rotting(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size(), fresh = 0, time = 0;
    queue<pair<int,int>> q;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 2) q.push({i, j});
            if (grid[i][j] == 1) fresh++;
        }
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    while (!q.empty() && fresh > 0) {
        int sz = q.size();
        while (sz--) {
            auto [x, y] = q.front(); q.pop();
            for (int d = 0; d < 4; d++) {
                int nx = x+dx[d], ny = y+dy[d];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    grid[nx][ny] = 2;
                    fresh--;
                    q.push({nx, ny});
                }
            }
        }
        time++;
    }
    return fresh == 0 ? time : -1;
}

// Word Ladder — BFS shortest transformation
int word_ladder(const string& beginWord, const string& endWord, const vector<string>& wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    if (!dict.count(endWord)) return 0;
    queue<pair<string, int>> q;
    q.push({beginWord, 1});
    dict.erase(beginWord);
    while (!q.empty()) {
        auto [word, steps] = q.front(); q.pop();
        if (word == endWord) return steps;
        for (int i = 0; i < (int)word.size(); i++) {
            char orig = word[i];
            for (char c = 'a'; c <= 'z'; c++) {
                word[i] = c;
                if (dict.count(word)) { dict.erase(word); q.push({word, steps + 1}); }
            }
            word[i] = orig;
        }
    }
    return 0;
}

// Surrounded Regions — BFS from border
void surrounded_regions(vector<vector<char>>& board) {
    int m = board.size(), n = board[0].size();
    queue<pair<int,int>> q;
    // Mark border-connected O's
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
        if ((i == 0 || i == m-1 || j == 0 || j == n-1) && board[i][j] == 'O') {
            q.push({i, j}); board[i][j] = 'S'; // safe
        }
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        for (int d = 0; d < 4; d++) {
            int nx = x+dx[d], ny = y+dy[d];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && board[nx][ny] == 'O') {
                board[nx][ny] = 'S'; q.push({nx, ny});
            }
        }
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 'O') board[i][j] = 'X';
            if (board[i][j] == 'S') board[i][j] = 'O';
        }
}

// Clone Graph
struct GraphNode {
    int val;
    vector<GraphNode*> neighbors;
    GraphNode(int v) : val(v) {}
};

GraphNode* clone_graph(GraphNode* node) {
    if (!node) return nullptr;
    unordered_map<GraphNode*, GraphNode*> cloned;
    queue<GraphNode*> q;
    q.push(node);
    cloned[node] = new GraphNode(node->val);
    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        for (auto nb : cur->neighbors) {
            if (!cloned.count(nb)) {
                cloned[nb] = new GraphNode(nb->val);
                q.push(nb);
            }
            cloned[cur]->neighbors.push_back(cloned[nb]);
        }
    }
    return cloned[node];
}

// Alien Dictionary — Topological sort on character order
string alien_order(const vector<string>& words) {
    unordered_map<char, unordered_set<char>> adj;
    unordered_map<char, int> indegree;
    for (auto& w : words) for (char c : w) { adj[c]; indegree[c]; }

    for (int i = 0; i + 1 < (int)words.size(); i++) {
        const string& a = words[i], &b = words[i+1];
        int len = min(a.size(), b.size());
        if (a.size() > b.size() && a.substr(0, len) == b.substr(0, len)) return ""; // invalid
        for (int j = 0; j < len; j++) {
            if (a[j] != b[j]) {
                if (!adj[a[j]].count(b[j])) { adj[a[j]].insert(b[j]); indegree[b[j]]++; }
                break;
            }
        }
    }

    queue<char> q;
    for (auto& [c, deg] : indegree) if (deg == 0) q.push(c);
    string result;
    while (!q.empty()) {
        char c = q.front(); q.pop();
        result += c;
        for (char nb : adj[c]) if (--indegree[nb] == 0) q.push(nb);
    }
    return result.size() == indegree.size() ? result : "";
}

// Cheapest Flights Within K Stops — modified Bellman-Ford
int cheapest_flights(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    for (int i = 0; i <= k; i++) {
        vector<int> tmp(dist);
        for (auto& f : flights) {
            int u = f[0], v = f[1], w = f[2];
            if (dist[u] != INT_MAX) tmp[v] = min(tmp[v], dist[u] + w);
        }
        dist = tmp;
    }
    return dist[dst] == INT_MAX ? -1 : dist[dst];
}

// Evaluate Division — weighted graph DFS
vector<double> calc_equation(vector<vector<string>>& equations, vector<double>& values,
                              vector<vector<string>>& queries) {
    unordered_map<string, vector<pair<string, double>>> g;
    for (int i = 0; i < (int)equations.size(); i++) {
        g[equations[i][0]].push_back({equations[i][1], values[i]});
        g[equations[i][1]].push_back({equations[i][0], 1.0 / values[i]});
    }

    auto bfs = [&](const string& src, const string& dst) -> double {
        if (!g.count(src) || !g.count(dst)) return -1.0;
        unordered_set<string> visited;
        queue<pair<string, double>> q;
        q.push({src, 1.0});
        visited.insert(src);
        while (!q.empty()) {
            auto [node, val] = q.front(); q.pop();
            if (node == dst) return val;
            for (auto& [next, w] : g[node]) {
                if (!visited.count(next)) {
                    visited.insert(next);
                    q.push({next, val * w});
                }
            }
        }
        return -1.0;
    };

    vector<double> result;
    for (auto& q : queries) result.push_back(bfs(q[0], q[1]));
    return result;
}

// ═══════════════════════════════════════════════════════════
//    P R E F I X   S U M   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Subarray Sum Equals K — O(N) with hashmap
int subarray_sum_k(const vector<int>& nums, int k) {
    unordered_map<int, int> prefix_count;
    prefix_count[0] = 1;
    int sum = 0, count = 0;
    for (int x : nums) {
        sum += x;
        if (prefix_count.count(sum - k)) count += prefix_count[sum - k];
        prefix_count[sum]++;
    }
    return count;
}

// Product of Array Except Self — O(N), no division
vector<int> product_except_self(const vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, 1);
    int prefix = 1;
    for (int i = 0; i < n; i++) { res[i] = prefix; prefix *= nums[i]; }
    int suffix = 1;
    for (int i = n-1; i >= 0; i--) { res[i] *= suffix; suffix *= nums[i]; }
    return res;
}

// Continuous Subarray Sum — sum divisible by k
bool check_subarray_sum(const vector<int>& nums, int k) {
    unordered_map<int, int> remainder_idx;
    remainder_idx[0] = -1;
    int sum = 0;
    for (int i = 0; i < (int)nums.size(); i++) {
        sum += nums[i];
        int rem = ((sum % k) + k) % k;
        if (remainder_idx.count(rem)) {
            if (i - remainder_idx[rem] >= 2) return true;
        } else {
            remainder_idx[rem] = i;
        }
    }
    return false;
}

// ═══════════════════════════════════════════════════════════
//     S A M P L I N G   A L G O R I T H M S
// ═══════════════════════════════════════════════════════════

mt19937 rng_sampling(chrono::steady_clock::now().time_since_epoch().count());

// Reservoir Sampling — pick k items uniformly from stream
vector<int> reservoir_sample(const vector<int>& stream, int k) {
    vector<int> reservoir(stream.begin(), stream.begin() + k);
    for (int i = k; i < (int)stream.size(); i++) {
        int j = uniform_int_distribution<int>(0, i)(rng_sampling);
        if (j < k) reservoir[j] = stream[i];
    }
    return reservoir;
}

// Fisher-Yates Shuffle — in-place uniform random permutation
void fisher_yates_shuffle(vector<int>& a) {
    for (int i = (int)a.size() - 1; i > 0; i--) {
        int j = uniform_int_distribution<int>(0, i)(rng_sampling);
        swap(a[i], a[j]);
    }
}

// Random Pick with Weight — prefix sum + binary search
struct WeightedRandom {
    vector<int> prefix;
    mt19937 rng{42};

    WeightedRandom(const vector<int>& w) : prefix(w.size()) {
        partial_sum(w.begin(), w.end(), prefix.begin());
    }

    int pickIndex() {
        int total = prefix.back();
        int target = uniform_int_distribution<int>(1, total)(rng);
        return (int)(lower_bound(prefix.begin(), prefix.end(), target) - prefix.begin());
    }
};

// ═══════════════════════════════════════════════════════════
//    L I N K E D   L I S T   P A T T E R N S
// ═══════════════════════════════════════════════════════════

struct ListNode {
    int val;
    ListNode* next;
    ListNode* random; // for Copy List with Random Pointer
    ListNode(int v) : val(v), next(nullptr), random(nullptr) {}
};

// Copy List with Random Pointer — O(N) time, O(1) extra space
ListNode* copy_random_list(ListNode* head) {
    if (!head) return nullptr;
    // Step 1: interleave clones
    for (auto cur = head; cur; ) {
        auto clone = new ListNode(cur->val);
        clone->next = cur->next;
        cur->next = clone;
        cur = clone->next;
    }
    // Step 2: assign random pointers
    for (auto cur = head; cur; cur = cur->next->next)
        if (cur->random) cur->next->random = cur->random->next;
    // Step 3: separate lists
    auto dummy = new ListNode(0);
    auto tail = dummy;
    for (auto cur = head; cur; ) {
        tail->next = cur->next;
        tail = tail->next;
        cur->next = cur->next->next;
        cur = cur->next;
    }
    return dummy->next;
}

// Reorder List: L0→L1→...→Ln → L0→Ln→L1→Ln-1→...
void reorder_list(ListNode* head) {
    if (!head || !head->next) return;
    // Find middle
    auto slow = head, fast = head;
    while (fast->next && fast->next->next) { slow = slow->next; fast = fast->next->next; }
    // Reverse second half
    auto prev = (ListNode*)nullptr, cur = slow->next;
    slow->next = nullptr;
    while (cur) { auto next = cur->next; cur->next = prev; prev = cur; cur = next; }
    // Merge alternately
    auto p1 = head, p2 = prev;
    while (p2) {
        auto n1 = p1->next, n2 = p2->next;
        p1->next = p2;
        p2->next = n1;
        p1 = n1;
        p2 = n2;
    }
}

// Reverse Nodes in K-Group
ListNode* reverse_k_group(ListNode* head, int k) {
    // Check if k nodes available
    auto check = head;
    for (int i = 0; i < k; i++) { if (!check) return head; check = check->next; }
    // Reverse k nodes
    auto prev = (ListNode*)nullptr, cur = head;
    for (int i = 0; i < k; i++) { auto next = cur->next; cur->next = prev; prev = cur; cur = next; }
    head->next = reverse_k_group(cur, k);
    return prev;
}

// Add Two Numbers (reverse order)
ListNode* add_two_numbers(ListNode* l1, ListNode* l2) {
    auto dummy = new ListNode(0);
    auto cur = dummy;
    int carry = 0;
    while (l1 || l2 || carry) {
        int sum = carry;
        if (l1) { sum += l1->val; l1 = l1->next; }
        if (l2) { sum += l2->val; l2 = l2->next; }
        carry = sum / 10;
        cur->next = new ListNode(sum % 10);
        cur = cur->next;
    }
    return dummy->next;
}

// ═══════════════════════════════════════════════════════════
//    B I N A R Y   S E A R C H   (FAANG patterns)
// ═══════════════════════════════════════════════════════════

// Koko Eating Bananas — BS on answer
int min_eating_speed(const vector<int>& piles, int h) {
    int lo = 1, hi = *max_element(piles.begin(), piles.end());
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        long long hours = 0;
        for (int p : piles) hours += (p + mid - 1) / mid;
        if (hours <= h) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Split Array Largest Sum — BS on answer
int split_array(const vector<int>& nums, int k) {
    int lo = *max_element(nums.begin(), nums.end());
    int hi = accumulate(nums.begin(), nums.end(), 0);
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int parts = 1, cur_sum = 0;
        for (int x : nums) {
            if (cur_sum + x > mid) { parts++; cur_sum = 0; }
            cur_sum += x;
        }
        if (parts <= k) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Find First and Last Position of Element in Sorted Array
pair<int,int> search_range(const vector<int>& nums, int target) {
    auto first = (int)(lower_bound(nums.begin(), nums.end(), target) - nums.begin());
    auto last = (int)(upper_bound(nums.begin(), nums.end(), target) - nums.begin()) - 1;
    if (first > last) return {-1, -1};
    return {first, last};
}

// ═══════════════════════════════════════════════════════════
//    T R I E   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Word Search II — Trie + grid DFS
struct TrieNode {
    TrieNode* ch[26] = {};
    string* word = nullptr; // store complete word at terminal
};

vector<string> word_search_2(vector<vector<char>>& board, vector<string>& words) {
    TrieNode root;
    for (auto& w : words) {
        auto* node = &root;
        for (char c : w) {
            if (!node->ch[c - 'a']) node->ch[c - 'a'] = new TrieNode();
            node = node->ch[c - 'a'];
        }
        node->word = &w;
    }
    int m = board.size(), n = board[0].size();
    vector<string> result;
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};

    function<void(int, int, TrieNode*)> dfs = [&](int r, int c, TrieNode* node) {
        if (r < 0 || r >= m || c < 0 || c >= n || board[r][c] == '#') return;
        char ch = board[r][c];
        if (!node->ch[ch - 'a']) return;
        node = node->ch[ch - 'a'];
        if (node->word) { result.push_back(*node->word); node->word = nullptr; } // dedup
        board[r][c] = '#';
        for (int d = 0; d < 4; d++) dfs(r + dx[d], c + dy[d], node);
        board[r][c] = ch;
    };

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            dfs(i, j, &root);
    return result;
}

// ═══════════════════════════════════════════════════════════
//    G R A P H   P A T T E R N S   (Interview)
// ═══════════════════════════════════════════════════════════

// Redundant Connection — find the edge that creates a cycle (DSU)
vector<int> find_redundant_connection(vector<vector<int>>& edges) {
    int n = edges.size();
    vector<int> parent(n + 1), rank_(n + 1, 0);
    iota(parent.begin(), parent.end(), 0);
    function<int(int)> find = [&](int x) -> int {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    for (auto& e : edges) {
        int a = find(e[0]), b = find(e[1]);
        if (a == b) return e;
        if (rank_[a] < rank_[b]) swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
    }
    return {};
}

// Accounts Merge — DSU on emails
vector<vector<string>> accounts_merge(vector<vector<string>>& accounts) {
    unordered_map<string, int> email_to_id;
    unordered_map<string, string> email_to_name;
    int id = 0;
    vector<int> parent(10001);
    iota(parent.begin(), parent.end(), 0);
    function<int(int)> find = [&](int x) -> int {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };

    for (auto& acc : accounts) {
        for (int i = 1; i < (int)acc.size(); i++) {
            email_to_name[acc[i]] = acc[0];
            if (!email_to_id.count(acc[i])) email_to_id[acc[i]] = id++;
            parent[find(email_to_id[acc[i]])] = find(email_to_id[acc[1]]);
        }
    }

    unordered_map<int, set<string>> groups;
    for (auto& [email, eid] : email_to_id)
        groups[find(eid)].insert(email);

    vector<vector<string>> result;
    for (auto& [root, emails] : groups) {
        vector<string> merged = {email_to_name[*emails.begin()]};
        for (auto& e : emails) merged.push_back(e);
        result.push_back(merged);
    }
    return result;
}
