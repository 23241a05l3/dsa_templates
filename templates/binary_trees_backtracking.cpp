/*
 * ============================================================
 *     BINARY TREES & BACKTRACKING — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *   BINARY TREES:
 *    1.  Traversals (In/Pre/Post/Level — iterative & recursive)
 *    2.  Height / Depth / Diameter
 *    3.  Check Balanced, Symmetric, Same Tree
 *    4.  LCA of Binary Tree (non-BST)
 *    5.  Path Sum, Max Path Sum
 *    6.  Serialize / Deserialize
 *    7.  Views (Left, Right, Top, Bottom)
 *    8.  Flatten BT to Linked List
 *    9.  Build from Inorder + Preorder / Postorder
 *   BACKTRACKING:
 *   10.  Subsets (with / without duplicates)
 *   11.  Permutations (with / without duplicates)
 *   12.  Combinations / Combination Sum
 *   13.  N-Queens
 *   14.  Sudoku Solver
 *   15.  Word Search
 *   16.  Generate Parentheses
 *   17.  Graph Coloring
 *   18.  Hamiltonian Path
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
//                    BINARY TREE NODE
// ═══════════════════════════════════════════════════════════
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

// ═══════════════════════════════════════════════════════════
// 1. TRAVERSALS — Iterative & Recursive
// ═══════════════════════════════════════════════════════════

// Inorder (Left, Root, Right) — Recursive
void inorder_rec(TreeNode* root, vector<int>& res) {
    if (!root) return;
    inorder_rec(root->left, res);
    res.push_back(root->val);
    inorder_rec(root->right, res);
}

// Inorder — Iterative (Morris Traversal — O(1) space)
vector<int> inorder_morris(TreeNode* root) {
    vector<int> res;
    TreeNode* cur = root;
    while (cur) {
        if (!cur->left) {
            res.push_back(cur->val);
            cur = cur->right;
        } else {
            TreeNode* pred = cur->left;
            while (pred->right && pred->right != cur)
                pred = pred->right;
            if (!pred->right) {
                pred->right = cur;
                cur = cur->left;
            } else {
                pred->right = nullptr;
                res.push_back(cur->val);
                cur = cur->right;
            }
        }
    }
    return res;
}

// Preorder — Iterative
vector<int> preorder_iter(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        auto node = st.top(); st.pop();
        res.push_back(node->val);
        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }
    return res;
}

// Postorder — Iterative (two stacks)
vector<int> postorder_iter(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    stack<TreeNode*> s1, s2;
    s1.push(root);
    while (!s1.empty()) {
        auto node = s1.top(); s1.pop();
        s2.push(node);
        if (node->left) s1.push(node->left);
        if (node->right) s1.push(node->right);
    }
    while (!s2.empty()) { res.push_back(s2.top()->val); s2.pop(); }
    return res;
}

// Level Order (BFS)
vector<vector<int>> level_order(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        vector<int> level;
        while (sz--) {
            auto node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        res.push_back(level);
    }
    return res;
}

// ═══════════════════════════════════════════════════════════
// 2. HEIGHT / DEPTH / DIAMETER
// ═══════════════════════════════════════════════════════════
int height(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(height(root->left), height(root->right));
}

int diameter_ans;
int diameter_helper(TreeNode* root) {
    if (!root) return 0;
    int l = diameter_helper(root->left);
    int r = diameter_helper(root->right);
    diameter_ans = max(diameter_ans, l + r);
    return 1 + max(l, r);
}
int diameter(TreeNode* root) {
    diameter_ans = 0;
    diameter_helper(root);
    return diameter_ans;
}

// ═══════════════════════════════════════════════════════════
// 3. CHECKS: Balanced, Symmetric, Same Tree
// ═══════════════════════════════════════════════════════════
int is_balanced_helper(TreeNode* root) {
    if (!root) return 0;
    int l = is_balanced_helper(root->left);
    int r = is_balanced_helper(root->right);
    if (l == -1 || r == -1 || abs(l - r) > 1) return -1;
    return 1 + max(l, r);
}
bool is_balanced(TreeNode* root) { return is_balanced_helper(root) != -1; }

bool is_symmetric_check(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && is_symmetric_check(a->left, b->right)
                            && is_symmetric_check(a->right, b->left);
}
bool is_symmetric(TreeNode* root) {
    return !root || is_symmetric_check(root->left, root->right);
}

bool is_same(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && is_same(a->left, b->left) && is_same(a->right, b->right);
}

// ═══════════════════════════════════════════════════════════
// 4. LCA OF BINARY TREE (not BST) — O(N)
// ═══════════════════════════════════════════════════════════
TreeNode* lca_bt(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    auto l = lca_bt(root->left, p, q);
    auto r = lca_bt(root->right, p, q);
    if (l && r) return root;
    return l ? l : r;
}

// ═══════════════════════════════════════════════════════════
// 5. PATH SUM / MAX PATH SUM
// ═══════════════════════════════════════════════════════════
bool has_path_sum(TreeNode* root, int target) {
    if (!root) return false;
    if (!root->left && !root->right) return target == root->val;
    return has_path_sum(root->left, target - root->val) ||
           has_path_sum(root->right, target - root->val);
}

int max_path_sum_val;
int max_path_sum_helper(TreeNode* root) {
    if (!root) return 0;
    int l = max(0, max_path_sum_helper(root->left));
    int r = max(0, max_path_sum_helper(root->right));
    max_path_sum_val = max(max_path_sum_val, l + r + root->val);
    return root->val + max(l, r);
}
int max_path_sum(TreeNode* root) {
    max_path_sum_val = INT_MIN;
    max_path_sum_helper(root);
    return max_path_sum_val;
}

// ═══════════════════════════════════════════════════════════
// 6. SERIALIZE / DESERIALIZE (Preorder with "#" for null)
// ═══════════════════════════════════════════════════════════
string serialize(TreeNode* root) {
    if (!root) return "#";
    return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
}

TreeNode* deserialize_helper(istringstream& ss) {
    string token;
    getline(ss, token, ',');
    if (token == "#") return nullptr;
    auto node = new TreeNode(stoi(token));
    node->left = deserialize_helper(ss);
    node->right = deserialize_helper(ss);
    return node;
}
TreeNode* deserialize(const string& data) {
    istringstream ss(data);
    return deserialize_helper(ss);
}

// ═══════════════════════════════════════════════════════════
// 7. VIEWS (Left, Right, Top, Bottom)
// ═══════════════════════════════════════════════════════════
vector<int> left_view(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            auto node = q.front(); q.pop();
            if (i == 0) res.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return res;
}

vector<int> right_view(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            auto node = q.front(); q.pop();
            if (i == sz - 1) res.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return res;
}

// Top view using horizontal distance
vector<int> top_view(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    map<int, int> mp; // hd -> val
    queue<pair<TreeNode*, int>> q;
    q.push({root, 0});
    while (!q.empty()) {
        auto [node, hd] = q.front(); q.pop();
        if (mp.find(hd) == mp.end()) mp[hd] = node->val;
        if (node->left) q.push({node->left, hd - 1});
        if (node->right) q.push({node->right, hd + 1});
    }
    for (auto& [k, v] : mp) res.push_back(v);
    return res;
}

// Bottom view using horizontal distance
vector<int> bottom_view(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    map<int, int> mp;
    queue<pair<TreeNode*, int>> q;
    q.push({root, 0});
    while (!q.empty()) {
        auto [node, hd] = q.front(); q.pop();
        mp[hd] = node->val; // overwrite each time — last one is bottom
        if (node->left) q.push({node->left, hd - 1});
        if (node->right) q.push({node->right, hd + 1});
    }
    for (auto& [k, v] : mp) res.push_back(v);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 8. FLATTEN BT TO LINKED LIST (in-place, preorder)
// ═══════════════════════════════════════════════════════════
void flatten(TreeNode* root) {
    TreeNode* cur = root;
    while (cur) {
        if (cur->left) {
            TreeNode* pred = cur->left;
            while (pred->right) pred = pred->right;
            pred->right = cur->right;
            cur->right = cur->left;
            cur->left = nullptr;
        }
        cur = cur->right;
    }
}

// ═══════════════════════════════════════════════════════════
// 9. BUILD FROM INORDER + PREORDER / POSTORDER
// ═══════════════════════════════════════════════════════════
TreeNode* build_pre_in(vector<int>& pre, vector<int>& in, int ps, int pe,
                       int is_, int ie, unordered_map<int,int>& idx) {
    if (ps > pe || is_ > ie) return nullptr;
    auto root = new TreeNode(pre[ps]);
    int ri = idx[pre[ps]];
    int left_sz = ri - is_;
    root->left = build_pre_in(pre, in, ps+1, ps+left_sz, is_, ri-1, idx);
    root->right = build_pre_in(pre, in, ps+left_sz+1, pe, ri+1, ie, idx);
    return root;
}
TreeNode* build_from_preorder_inorder(vector<int>& pre, vector<int>& inorder) {
    unordered_map<int,int> idx;
    for (int i = 0; i < (int)inorder.size(); i++) idx[inorder[i]] = i;
    return build_pre_in(pre, inorder, 0, pre.size()-1, 0, inorder.size()-1, idx);
}

TreeNode* build_post_in(vector<int>& post, vector<int>& in, int ps, int pe,
                        int is_, int ie, unordered_map<int,int>& idx) {
    if (ps > pe || is_ > ie) return nullptr;
    auto root = new TreeNode(post[pe]);
    int ri = idx[post[pe]];
    int left_sz = ri - is_;
    root->left = build_post_in(post, in, ps, ps+left_sz-1, is_, ri-1, idx);
    root->right = build_post_in(post, in, ps+left_sz, pe-1, ri+1, ie, idx);
    return root;
}
TreeNode* build_from_postorder_inorder(vector<int>& post, vector<int>& inorder) {
    unordered_map<int,int> idx;
    for (int i = 0; i < (int)inorder.size(); i++) idx[inorder[i]] = i;
    return build_post_in(post, inorder, 0, post.size()-1, 0, inorder.size()-1, idx);
}

// ═══════════════════════════════════════════════════════════
//             B A C K T R A C K I N G
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// 10. SUBSETS — with / without duplicates
// ═══════════════════════════════════════════════════════════
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> cur;
    function<void(int)> bt = [&](int i) {
        res.push_back(cur);
        for (int j = i; j < (int)nums.size(); j++) {
            cur.push_back(nums[j]);
            bt(j + 1);
            cur.pop_back();
        }
    };
    bt(0);
    return res;
}

// With duplicates (sort first)
vector<vector<int>> subsets_dup(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    vector<int> cur;
    function<void(int)> bt = [&](int i) {
        res.push_back(cur);
        for (int j = i; j < (int)nums.size(); j++) {
            if (j > i && nums[j] == nums[j-1]) continue; // skip dup
            cur.push_back(nums[j]);
            bt(j + 1);
            cur.pop_back();
        }
    };
    bt(0);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 11. PERMUTATIONS — with / without duplicates
// ═══════════════════════════════════════════════════════════
vector<vector<int>> permutations(vector<int>& nums) {
    vector<vector<int>> res;
    function<void(int)> bt = [&](int start) {
        if (start == (int)nums.size()) { res.push_back(nums); return; }
        for (int i = start; i < (int)nums.size(); i++) {
            swap(nums[start], nums[i]);
            bt(start + 1);
            swap(nums[start], nums[i]);
        }
    };
    bt(0);
    return res;
}

// With duplicates — use used[] array
vector<vector<int>> permutations_dup(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    vector<int> cur;
    vector<bool> used(nums.size(), false);
    function<void()> bt = [&]() {
        if (cur.size() == nums.size()) { res.push_back(cur); return; }
        for (int i = 0; i < (int)nums.size(); i++) {
            if (used[i]) continue;
            if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;
            used[i] = true;
            cur.push_back(nums[i]);
            bt();
            cur.pop_back();
            used[i] = false;
        }
    };
    bt();
    return res;
}

// ═══════════════════════════════════════════════════════════
// 12. COMBINATIONS / COMBINATION SUM
// ═══════════════════════════════════════════════════════════
// C(n, k)
vector<vector<int>> combinations(int n, int k) {
    vector<vector<int>> res;
    vector<int> cur;
    function<void(int)> bt = [&](int start) {
        if ((int)cur.size() == k) { res.push_back(cur); return; }
        for (int i = start; i <= n - (k - (int)cur.size()) + 1; i++) {
            cur.push_back(i);
            bt(i + 1);
            cur.pop_back();
        }
    };
    bt(1);
    return res;
}

// Combination Sum — unlimited use of each candidate
vector<vector<int>> combination_sum(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> res;
    vector<int> cur;
    function<void(int, int)> bt = [&](int start, int remain) {
        if (remain == 0) { res.push_back(cur); return; }
        for (int i = start; i < (int)candidates.size() && candidates[i] <= remain; i++) {
            cur.push_back(candidates[i]);
            bt(i, remain - candidates[i]); // same element reusable
            cur.pop_back();
        }
    };
    bt(0, target);
    return res;
}

// Combination Sum II — each candidate used at most once
vector<vector<int>> combination_sum2(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> res;
    vector<int> cur;
    function<void(int, int)> bt = [&](int start, int remain) {
        if (remain == 0) { res.push_back(cur); return; }
        for (int i = start; i < (int)candidates.size() && candidates[i] <= remain; i++) {
            if (i > start && candidates[i] == candidates[i-1]) continue;
            cur.push_back(candidates[i]);
            bt(i + 1, remain - candidates[i]);
            cur.pop_back();
        }
    };
    bt(0, target);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 13. N-QUEENS
// ═══════════════════════════════════════════════════════════
vector<vector<string>> solve_n_queens(int n) {
    vector<vector<string>> res;
    vector<string> board(n, string(n, '.'));
    vector<bool> col(n), diag1(2*n), diag2(2*n); // col, main diag, anti diag

    function<void(int)> bt = [&](int row) {
        if (row == n) { res.push_back(board); return; }
        for (int c = 0; c < n; c++) {
            if (col[c] || diag1[row-c+n] || diag2[row+c]) continue;
            board[row][c] = 'Q';
            col[c] = diag1[row-c+n] = diag2[row+c] = true;
            bt(row + 1);
            board[row][c] = '.';
            col[c] = diag1[row-c+n] = diag2[row+c] = false;
        }
    };
    bt(0);
    return res;
}

// Count solutions only
int count_n_queens(int n) {
    int cnt = 0;
    vector<bool> col(n), diag1(2*n), diag2(2*n);
    function<void(int)> bt = [&](int row) {
        if (row == n) { cnt++; return; }
        for (int c = 0; c < n; c++) {
            if (col[c] || diag1[row-c+n] || diag2[row+c]) continue;
            col[c] = diag1[row-c+n] = diag2[row+c] = true;
            bt(row + 1);
            col[c] = diag1[row-c+n] = diag2[row+c] = false;
        }
    };
    bt(0);
    return cnt;
}

// ═══════════════════════════════════════════════════════════
// 14. SUDOKU SOLVER
// ═══════════════════════════════════════════════════════════
bool solve_sudoku(vector<vector<char>>& board) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] != '.') continue;
            for (char c = '1'; c <= '9'; c++) {
                // Check validity
                bool valid = true;
                for (int k = 0; k < 9 && valid; k++) {
                    if (board[i][k] == c) valid = false;
                    if (board[k][j] == c) valid = false;
                    if (board[3*(i/3)+k/3][3*(j/3)+k%3] == c) valid = false;
                }
                if (valid) {
                    board[i][j] = c;
                    if (solve_sudoku(board)) return true;
                    board[i][j] = '.';
                }
            }
            return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════
// 15. WORD SEARCH (Word exists in grid?)
// ═══════════════════════════════════════════════════════════
bool word_search(vector<vector<char>>& board, const string& word) {
    int m = board.size(), n = board[0].size();
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    function<bool(int,int,int)> dfs = [&](int r, int c, int idx) -> bool {
        if (idx == (int)word.size()) return true;
        if (r < 0 || r >= m || c < 0 || c >= n || board[r][c] != word[idx]) return false;
        char tmp = board[r][c];
        board[r][c] = '#'; // mark visited
        for (int d = 0; d < 4; d++)
            if (dfs(r+dx[d], c+dy[d], idx+1)) return true;
        board[r][c] = tmp;
        return false;
    };
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (dfs(i, j, 0)) return true;
    return false;
}

// ═══════════════════════════════════════════════════════════
// 16. GENERATE PARENTHESES
// ═══════════════════════════════════════════════════════════
vector<string> generate_parentheses(int n) {
    vector<string> res;
    string cur;
    function<void(int, int)> bt = [&](int open, int close) {
        if ((int)cur.size() == 2 * n) { res.push_back(cur); return; }
        if (open < n) { cur += '('; bt(open + 1, close); cur.pop_back(); }
        if (close < open) { cur += ')'; bt(open, close + 1); cur.pop_back(); }
    };
    bt(0, 0);
    return res;
}

// ═══════════════════════════════════════════════════════════
// 17. GRAPH COLORING — color graph with m colors
// ═══════════════════════════════════════════════════════════
bool graph_coloring(vector<vector<int>>& adj, int n, int m, vector<int>& color) {
    // color[i] = 0 means uncolored, colors are 1..m
    color.assign(n, 0);
    function<bool(int)> bt = [&](int node) -> bool {
        if (node == n) return true;
        for (int c = 1; c <= m; c++) {
            bool ok = true;
            for (int nb : adj[node])
                if (color[nb] == c) { ok = false; break; }
            if (ok) {
                color[node] = c;
                if (bt(node + 1)) return true;
                color[node] = 0;
            }
        }
        return false;
    };
    return bt(0);
}

// ═══════════════════════════════════════════════════════════
// 18. HAMILTONIAN PATH (visit all vertices exactly once)
// ═══════════════════════════════════════════════════════════
bool hamiltonian_path(vector<vector<int>>& adj, int n, vector<int>& path) {
    vector<bool> vis(n, false);
    path.clear();

    function<bool(int)> bt = [&](int u) -> bool {
        path.push_back(u);
        if ((int)path.size() == n) return true;
        vis[u] = true;
        for (int v : adj[u]) {
            if (!vis[v] && bt(v)) return true;
        }
        vis[u] = false;
        path.pop_back();
        return false;
    };

    for (int i = 0; i < n; i++)
        if (bt(i)) return true;
    return false;
}

/*
 * ══════════════════════════════════════
 *  BACKTRACKING TIPS:
 * ══════════════════════════════════════
 *
 * 1. Choose → Explore → Un-choose (the core pattern)
 * 2. Sort first when skipping duplicates
 * 3. Use pruning aggressively (e.g., if remaining sum < target, break)
 * 4. For grid problems: mark visited in-place (temp replacement)
 * 5. For constraint satisfaction (Sudoku, N-Queens):
 *    use bitmask for O(1) validity checks
 * 6. Time complexity: usually exponential, prune to manage
 */
