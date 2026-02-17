/*
 * ============================================================
 *     INTERVIEW DP PATTERNS — FAANG TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Kadane's Algorithm (Max Subarray)
 *    2.  Maximum Product Subarray
 *    3.  Stock Buy/Sell I–IV + Cooldown + Fee
 *    4.  House Robber (Linear / Circular / Tree)
 *    5.  Word Break (I & II)
 *    6.  Decode Ways
 *    7.  Palindrome Partitioning (all + min cut)
 *    8.  Longest Palindromic Subsequence
 *    9.  Jump Game (I & II)
 *   10.  Interleaving Strings
 *   11.  Wildcard / Regex Matching
 *   12.  Distinct Subsequences
 *   13.  Maximal Square
 *   14.  Paint House / K Colors
 *   15.  Ugly Numbers
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
// 1. KADANE'S ALGORITHM — Maximum Subarray Sum — O(N)
// ═══════════════════════════════════════════════════════════
int max_subarray(const vector<int>& a) {
    int cur = a[0], best = a[0];
    for (int i = 1; i < (int)a.size(); i++) {
        cur = max(a[i], cur + a[i]);
        best = max(best, cur);
    }
    return best;
}

// Variant: return [l, r] indices of max subarray
tuple<int, int, int> max_subarray_range(const vector<int>& a) {
    int cur = a[0], best = a[0], start = 0, best_l = 0, best_r = 0;
    for (int i = 1; i < (int)a.size(); i++) {
        if (a[i] > cur + a[i]) { cur = a[i]; start = i; }
        else cur += a[i];
        if (cur > best) { best = cur; best_l = start; best_r = i; }
    }
    return {best, best_l, best_r};
}

// Max circular subarray sum
int max_circular_subarray(const vector<int>& a) {
    int total = 0, max_cur = a[0], max_sum = a[0], min_cur = a[0], min_sum = a[0];
    for (int i = 0; i < (int)a.size(); i++) total += a[i];
    for (int i = 1; i < (int)a.size(); i++) {
        max_cur = max(a[i], max_cur + a[i]);
        max_sum = max(max_sum, max_cur);
        min_cur = min(a[i], min_cur + a[i]);
        min_sum = min(min_sum, min_cur);
    }
    // If all negative, max_sum is the answer (can't wrap around with all negatives)
    return (total == min_sum) ? max_sum : max(max_sum, total - min_sum);
}

// ═══════════════════════════════════════════════════════════
// 2. MAXIMUM PRODUCT SUBARRAY — O(N)
// ═══════════════════════════════════════════════════════════
int max_product_subarray(const vector<int>& a) {
    int cur_max = a[0], cur_min = a[0], result = a[0];
    for (int i = 1; i < (int)a.size(); i++) {
        if (a[i] < 0) swap(cur_max, cur_min); // negative flips max/min
        cur_max = max(a[i], cur_max * a[i]);
        cur_min = min(a[i], cur_min * a[i]);
        result = max(result, cur_max);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
// 3. STOCK BUY/SELL — All Variants
// ═══════════════════════════════════════════════════════════

// I: At most 1 transaction
int stock_1(const vector<int>& prices) {
    int mn = INT_MAX, profit = 0;
    for (int p : prices) {
        mn = min(mn, p);
        profit = max(profit, p - mn);
    }
    return profit;
}

// II: Unlimited transactions
int stock_unlimited(const vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < (int)prices.size(); i++)
        profit += max(0, prices[i] - prices[i-1]);
    return profit;
}

// III: At most 2 transactions
int stock_2(const vector<int>& prices) {
    int buy1 = INT_MIN, sell1 = 0, buy2 = INT_MIN, sell2 = 0;
    for (int p : prices) {
        buy1 = max(buy1, -p);
        sell1 = max(sell1, buy1 + p);
        buy2 = max(buy2, sell1 - p);
        sell2 = max(sell2, buy2 + p);
    }
    return sell2;
}

// IV: At most k transactions — O(Nk) time, O(k) space
int stock_k(int k, const vector<int>& prices) {
    int n = prices.size();
    if (n == 0) return 0;
    if (k >= n / 2) return stock_unlimited(prices); // optimization
    vector<int> buy(k + 1, INT_MIN), sell(k + 1, 0);
    for (int p : prices)
        for (int j = 1; j <= k; j++) {
            buy[j] = max(buy[j], sell[j-1] - p);
            sell[j] = max(sell[j], buy[j] + p);
        }
    return sell[k];
}

// With Cooldown: after selling, must wait 1 day
int stock_cooldown(const vector<int>& prices) {
    int n = prices.size();
    if (n < 2) return 0;
    // States: hold, sold (just sold, cooldown next), rest (can buy)
    int hold = -prices[0], sold = 0, rest = 0;
    for (int i = 1; i < n; i++) {
        int prev_hold = hold, prev_sold = sold, prev_rest = rest;
        hold = max(prev_hold, prev_rest - prices[i]);
        sold = prev_hold + prices[i];
        rest = max(prev_rest, prev_sold);
    }
    return max(sold, rest);
}

// With Transaction Fee
int stock_fee(const vector<int>& prices, int fee) {
    int hold = -prices[0], cash = 0;
    for (int i = 1; i < (int)prices.size(); i++) {
        hold = max(hold, cash - prices[i]);
        cash = max(cash, hold + prices[i] - fee);
    }
    return cash;
}

// ═══════════════════════════════════════════════════════════
// 4. HOUSE ROBBER — Linear / Circular / Tree
// ═══════════════════════════════════════════════════════════

// Linear
int rob_linear(const vector<int>& nums) {
    int prev2 = 0, prev1 = 0;
    for (int x : nums) {
        int cur = max(prev1, prev2 + x);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

// Circular (first and last are adjacent)
int rob_circular(const vector<int>& nums) {
    int n = nums.size();
    if (n == 1) return nums[0];
    auto rob_range = [&](int l, int r) {
        int prev2 = 0, prev1 = 0;
        for (int i = l; i <= r; i++) {
            int cur = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = cur;
        }
        return prev1;
    };
    return max(rob_range(0, n-2), rob_range(1, n-1));
}

// Tree (using TreeNode from design_ds or binary_trees_backtracking)
struct TNode { int val; TNode *left, *right; };

pair<int,int> rob_tree_helper(TNode* root) {
    if (!root) return {0, 0}; // {rob this, skip this}
    auto [rob_l, skip_l] = rob_tree_helper(root->left);
    auto [rob_r, skip_r] = rob_tree_helper(root->right);
    int rob_this = root->val + skip_l + skip_r;
    int skip_this = max(rob_l, skip_l) + max(rob_r, skip_r);
    return {rob_this, skip_this};
}
int rob_tree(TNode* root) {
    auto [rob, skip] = rob_tree_helper(root);
    return max(rob, skip);
}

// ═══════════════════════════════════════════════════════════
// 5. WORD BREAK — I & II
// ═══════════════════════════════════════════════════════════

// I: Can s be segmented into words from dict?
bool word_break(const string& s, const vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = s.size();
    vector<bool> dp(n + 1, false);
    dp[0] = true;
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < i; j++)
            if (dp[j] && dict.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
    return dp[n];
}

// II: Return all possible sentences
vector<string> word_break_2(const string& s, const vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = s.size();
    vector<vector<string>> memo(n + 1);
    vector<int> computed(n + 1, -1); // -1 = not computed

    function<vector<string>(int)> bt = [&](int start) -> vector<string> {
        if (start == n) return {""};
        if (computed[start] != -1) return memo[start];
        computed[start] = 1;
        for (int end = start + 1; end <= n; end++) {
            string word = s.substr(start, end - start);
            if (dict.count(word)) {
                auto rest = bt(end);
                for (auto& r : rest)
                    memo[start].push_back(word + (r.empty() ? "" : " " + r));
            }
        }
        return memo[start];
    };
    return bt(0);
}

// ═══════════════════════════════════════════════════════════
// 6. DECODE WAYS — "226" → "2|2|6", "22|6", "2|26" → 3 ways
// ═══════════════════════════════════════════════════════════
int decode_ways(const string& s) {
    int n = s.size();
    if (n == 0 || s[0] == '0') return 0;
    int prev2 = 1, prev1 = 1; // dp[0] = 1, dp[1] = 1
    for (int i = 2; i <= n; i++) {
        int cur = 0;
        int one_digit = s[i-1] - '0';
        int two_digit = (s[i-2] - '0') * 10 + one_digit;
        if (one_digit >= 1) cur += prev1;
        if (two_digit >= 10 && two_digit <= 26) cur += prev2;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

// ═══════════════════════════════════════════════════════════
// 7. PALINDROME PARTITIONING — Enumerate All + Min Cut
// ═══════════════════════════════════════════════════════════

// Enumerate all palindromic partitions
vector<vector<string>> palindrome_partition(const string& s) {
    int n = s.size();
    vector<vector<bool>> is_pal(n, vector<bool>(n, false));
    for (int i = n-1; i >= 0; i--)
        for (int j = i; j < n; j++)
            is_pal[i][j] = (s[i] == s[j]) && (j - i <= 2 || is_pal[i+1][j-1]);

    vector<vector<string>> res;
    vector<string> cur;
    function<void(int)> bt = [&](int start) {
        if (start == n) { res.push_back(cur); return; }
        for (int end = start; end < n; end++) {
            if (is_pal[start][end]) {
                cur.push_back(s.substr(start, end - start + 1));
                bt(end + 1);
                cur.pop_back();
            }
        }
    };
    bt(0);
    return res;
}

// Minimum cuts for palindrome partitioning
int min_palindrome_cut(const string& s) {
    int n = s.size();
    vector<vector<bool>> is_pal(n, vector<bool>(n, false));
    for (int i = n-1; i >= 0; i--)
        for (int j = i; j < n; j++)
            is_pal[i][j] = (s[i] == s[j]) && (j - i <= 2 || is_pal[i+1][j-1]);

    vector<int> dp(n, INT_MAX); // dp[i] = min cuts for s[0..i]
    for (int i = 0; i < n; i++) {
        if (is_pal[0][i]) { dp[i] = 0; continue; }
        for (int j = 1; j <= i; j++)
            if (is_pal[j][i]) dp[i] = min(dp[i], dp[j-1] + 1);
    }
    return dp[n-1];
}

// ═══════════════════════════════════════════════════════════
// 8. LONGEST PALINDROMIC SUBSEQUENCE — O(N²)
// ═══════════════════════════════════════════════════════════
int longest_palindromic_subseq(const string& s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = n-1; i >= 0; i--) {
        dp[i][i] = 1;
        for (int j = i+1; j < n; j++) {
            if (s[i] == s[j]) dp[i][j] = dp[i+1][j-1] + 2;
            else dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
        }
    }
    return dp[0][n-1];
}

// ═══════════════════════════════════════════════════════════
// 9. JUMP GAME — I (can reach?) & II (min jumps)
// ═══════════════════════════════════════════════════════════

// I: Can you reach the last index?
bool can_jump(const vector<int>& nums) {
    int farthest = 0;
    for (int i = 0; i <= farthest && i < (int)nums.size(); i++)
        farthest = max(farthest, i + nums[i]);
    return farthest >= (int)nums.size() - 1;
}

// II: Minimum number of jumps to reach end — O(N) greedy
int min_jumps(const vector<int>& nums) {
    int jumps = 0, cur_end = 0, farthest = 0;
    for (int i = 0; i < (int)nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == cur_end) { jumps++; cur_end = farthest; }
    }
    return jumps;
}

// ═══════════════════════════════════════════════════════════
// 10. INTERLEAVING STRINGS — O(N*M)
// ═══════════════════════════════════════════════════════════
bool is_interleave(const string& s1, const string& s2, const string& s3) {
    int n = s1.size(), m = s2.size();
    if (n + m != (int)s3.size()) return false;
    vector<vector<bool>> dp(n+1, vector<bool>(m+1, false));
    dp[0][0] = true;
    for (int i = 1; i <= n; i++) dp[i][0] = dp[i-1][0] && s1[i-1] == s3[i-1];
    for (int j = 1; j <= m; j++) dp[0][j] = dp[0][j-1] && s2[j-1] == s3[j-1];
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[i+j-1]) ||
                        (dp[i][j-1] && s2[j-1] == s3[i+j-1]);
    return dp[n][m];
}

// ═══════════════════════════════════════════════════════════
// 11. WILDCARD MATCHING & REGEX MATCHING
// ═══════════════════════════════════════════════════════════

// Wildcard: '?' matches one char, '*' matches any sequence
bool wildcard_match(const string& s, const string& p) {
    int n = s.size(), m = p.size();
    vector<vector<bool>> dp(n+1, vector<bool>(m+1, false));
    dp[0][0] = true;
    for (int j = 1; j <= m; j++) dp[0][j] = dp[0][j-1] && p[j-1] == '*';
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            if (p[j-1] == '*') dp[i][j] = dp[i-1][j] || dp[i][j-1];
            else if (p[j-1] == '?' || s[i-1] == p[j-1]) dp[i][j] = dp[i-1][j-1];
        }
    return dp[n][m];
}

// Regex: '.' matches one char, '*' matches zero+ of preceding element
bool regex_match(const string& s, const string& p) {
    int n = s.size(), m = p.size();
    vector<vector<bool>> dp(n+1, vector<bool>(m+1, false));
    dp[0][0] = true;
    for (int j = 2; j <= m; j++)
        if (p[j-1] == '*') dp[0][j] = dp[0][j-2];
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            if (p[j-1] == '*') {
                dp[i][j] = dp[i][j-2]; // zero occurrences
                if (p[j-2] == '.' || p[j-2] == s[i-1])
                    dp[i][j] = dp[i][j] || dp[i-1][j]; // one+ occurrences
            } else if (p[j-1] == '.' || p[j-1] == s[i-1]) {
                dp[i][j] = dp[i-1][j-1];
            }
        }
    return dp[n][m];
}

// ═══════════════════════════════════════════════════════════
// 12. DISTINCT SUBSEQUENCES — count subseq of s equal to t
// ═══════════════════════════════════════════════════════════
int num_distinct(const string& s, const string& t) {
    int n = s.size(), m = t.size();
    vector<long long> dp(m + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++)
        for (int j = m; j >= 1; j--)  // reverse to avoid overwriting
            if (s[i-1] == t[j-1]) dp[j] += dp[j-1];
    return dp[m];
}

// ═══════════════════════════════════════════════════════════
// 13. MAXIMAL SQUARE — largest all-1s square in binary matrix
// ═══════════════════════════════════════════════════════════
int maximal_square(const vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size(), max_side = 0;
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == 1) {
                dp[i][j] = (i && j) ? min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1 : 1;
                max_side = max(max_side, dp[i][j]);
            }
        }
    return max_side * max_side;
}

// ═══════════════════════════════════════════════════════════
// 14. PAINT HOUSE — min cost, no two adjacent same color
// ═══════════════════════════════════════════════════════════

// K colors — O(NK) using two-min optimization
int paint_house(const vector<vector<int>>& costs) {
    int n = costs.size(), k = costs[0].size();
    // Track minimum and second minimum of previous row
    int min1 = 0, min2 = 0, min1_idx = -1;
    for (int i = 0; i < n; i++) {
        int new_min1 = INT_MAX, new_min2 = INT_MAX, new_min1_idx = -1;
        for (int j = 0; j < k; j++) {
            int prev = (j == min1_idx) ? min2 : min1;
            int cur = costs[i][j] + prev;
            if (cur < new_min1) { new_min2 = new_min1; new_min1 = cur; new_min1_idx = j; }
            else if (cur < new_min2) new_min2 = cur;
        }
        min1 = new_min1; min2 = new_min2; min1_idx = new_min1_idx;
    }
    return min1;
}

// ═══════════════════════════════════════════════════════════
// 15. UGLY NUMBERS — numbers whose prime factors are only 2, 3, 5
// ═══════════════════════════════════════════════════════════
int nth_ugly(int n) {
    vector<int> ugly(n);
    ugly[0] = 1;
    int i2 = 0, i3 = 0, i5 = 0;
    for (int i = 1; i < n; i++) {
        int next2 = ugly[i2] * 2, next3 = ugly[i3] * 3, next5 = ugly[i5] * 5;
        ugly[i] = min({next2, next3, next5});
        if (ugly[i] == next2) i2++;
        if (ugly[i] == next3) i3++;
        if (ugly[i] == next5) i5++;
    }
    return ugly[n-1];
}

// Super Ugly: given list of primes
int nth_super_ugly(int n, const vector<int>& primes) {
    int k = primes.size();
    vector<int> ugly(n), idx(k, 0);
    ugly[0] = 1;
    for (int i = 1; i < n; i++) {
        ugly[i] = INT_MAX;
        for (int j = 0; j < k; j++)
            ugly[i] = min(ugly[i], ugly[idx[j]] * primes[j]);
        for (int j = 0; j < k; j++)
            if (ugly[i] == ugly[idx[j]] * primes[j]) idx[j]++;
    }
    return ugly[n-1];
}

/*
 * ══════════════════════════════════════
 *  FAANG DP CHEAT SHEET:
 * ══════════════════════════════════════
 *
 * 1D DP:    Kadane, House Robber, Jump Game, Decode Ways, Climbing Stairs
 * 2D DP:    LCS, Edit Distance, Interleaving, Wildcard, Regex, Distinct Subseq
 * Interval: Palindrome Partition, Matrix Chain, Burst Balloons
 * State:    Stock problems (hold/sold/rest states)
 * Grid:     Maximal Square, Unique Paths, Min Path Sum
 * Tree DP:  House Robber III, Diameter, Max Path Sum
 * String:   Word Break, Palindrome checks
 *
 * Space optimization: roll 2 rows for 2D DP, or 2 variables for 1D
 */
