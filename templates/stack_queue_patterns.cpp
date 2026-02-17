/*
 * ============================================================
 *     STACK & QUEUE PATTERNS — FAANG TEMPLATE
 *     (Monotonic Stack, Calculators, Brackets, Decode,
 *      Sliding Window Max, Priority Queue patterns)
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
//     M O N O T O N I C   S T A C K
// ═══════════════════════════════════════════════════════════

// Next Greater Element — O(N)
vector<int> next_greater(const vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, -1);
    stack<int> st; // indices
    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            res[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    return res;
}

// Next Smaller Element
vector<int> next_smaller(const vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, -1);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] > nums[i]) {
            res[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    return res;
}

// Previous Greater / Previous Smaller
vector<int> prev_greater(const vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, -1);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] <= nums[i]) st.pop();
        if (!st.empty()) res[i] = nums[st.top()];
        st.push(i);
    }
    return res;
}

// ───────────────────────────────────────────────────────────
// Largest Rectangle in Histogram — O(N)
int largest_rect_histogram(const vector<int>& heights) {
    int n = heights.size(), maxArea = 0;
    stack<int> st;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()]; st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        st.push(i);
    }
    return maxArea;
}

// Maximal Rectangle in Binary Matrix — O(M*N)
int maximal_rectangle(const vector<vector<char>>& matrix) {
    int m = matrix.size(), n = matrix[0].size(), maxArea = 0;
    vector<int> heights(n, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            heights[j] = matrix[i][j] == '1' ? heights[j] + 1 : 0;
        maxArea = max(maxArea, largest_rect_histogram(heights));
    }
    return maxArea;
}

// Daily Temperatures — days to wait for warmer temp
vector<int> daily_temperatures(const vector<int>& temps) {
    int n = temps.size();
    vector<int> res(n, 0);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && temps[st.top()] < temps[i]) {
            res[st.top()] = i - st.top();
            st.pop();
        }
        st.push(i);
    }
    return res;
}

// Stock Span — how many consecutive days price was ≤ today
vector<int> stock_span(const vector<int>& prices) {
    int n = prices.size();
    vector<int> span(n);
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && prices[st.top()] <= prices[i]) st.pop();
        span[i] = st.empty() ? i + 1 : i - st.top();
        st.push(i);
    }
    return span;
}

// Trapping Rain Water — monotonic stack approach
int trap_rain_water(const vector<int>& height) {
    int n = height.size(), water = 0;
    stack<int> st;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && height[st.top()] < height[i]) {
            int bottom = height[st.top()]; st.pop();
            if (st.empty()) break;
            int width = i - st.top() - 1;
            int h = min(height[i], height[st.top()]) - bottom;
            water += width * h;
        }
        st.push(i);
    }
    return water;
}

// ═══════════════════════════════════════════════════════════
//     C A L C U L A T O R S   &   E X P R E S S I O N S
// ═══════════════════════════════════════════════════════════

// Basic Calculator I — handles +, -, (, )
int basic_calculator_1(const string& s) {
    stack<int> nums, ops;
    int result = 0, num = 0, sign = 1;
    for (char c : s) {
        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == '+' || c == '-') {
            result += sign * num;
            num = 0;
            sign = (c == '+') ? 1 : -1;
        } else if (c == '(') {
            nums.push(result);
            ops.push(sign);
            result = 0;
            sign = 1;
        } else if (c == ')') {
            result += sign * num;
            num = 0;
            result = nums.top() + ops.top() * result;
            nums.pop(); ops.pop();
        }
    }
    return result + sign * num;
}

// Basic Calculator II — handles +, -, *, /
int basic_calculator_2(const string& s) {
    stack<int> st;
    int num = 0;
    char op = '+';
    for (int i = 0; i <= (int)s.size(); i++) {
        char c = (i < (int)s.size()) ? s[i] : '+';
        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c != ' ') {
            if (op == '+') st.push(num);
            else if (op == '-') st.push(-num);
            else if (op == '*') { int t = st.top(); st.pop(); st.push(t * num); }
            else if (op == '/') { int t = st.top(); st.pop(); st.push(t / num); }
            op = c;
            num = 0;
        }
    }
    int result = 0;
    while (!st.empty()) { result += st.top(); st.pop(); }
    return result;
}

// Basic Calculator III — handles +, -, *, /, (, )
int basic_calculator_3(const string& s) {
    int i = 0;
    function<int()> parse = [&]() -> int {
        stack<int> st;
        int num = 0;
        char op = '+';
        while (i < (int)s.size()) {
            char c = s[i];
            if (isdigit(c)) {
                num = num * 10 + (c - '0');
            } else if (c == '(') {
                i++; // skip '('
                num = parse();
            }
            if ((!isdigit(c) && c != ' ') || i == (int)s.size() - 1) {
                if (op == '+') st.push(num);
                else if (op == '-') st.push(-num);
                else if (op == '*') { int t = st.top(); st.pop(); st.push(t * num); }
                else if (op == '/') { int t = st.top(); st.pop(); st.push(t / num); }
                op = c;
                num = 0;
            }
            if (c == ')') { i++; break; }
            i++;
        }
        int result = 0;
        while (!st.empty()) { result += st.top(); st.pop(); }
        return result;
    };
    return parse();
}

// Evaluate Reverse Polish Notation
int eval_rpn(const vector<string>& tokens) {
    stack<int> st;
    for (auto& t : tokens) {
        if (t == "+" || t == "-" || t == "*" || t == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();
            if (t == "+") st.push(a + b);
            else if (t == "-") st.push(a - b);
            else if (t == "*") st.push(a * b);
            else st.push(a / b);
        } else {
            st.push(stoi(t));
        }
    }
    return st.top();
}

// ═══════════════════════════════════════════════════════════
//     P A R E N T H E S E S   &   B R A C K E T S
// ═══════════════════════════════════════════════════════════

// Valid Parentheses — (), {}, []
bool is_valid_parens(const string& s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '{' || c == '[') st.push(c);
        else {
            if (st.empty()) return false;
            char top = st.top(); st.pop();
            if ((c == ')' && top != '(') || (c == '}' && top != '{') || (c == ']' && top != '['))
                return false;
        }
    }
    return st.empty();
}

// Longest Valid Parentheses — O(N)
int longest_valid_parens(const string& s) {
    stack<int> st;
    st.push(-1);
    int maxLen = 0;
    for (int i = 0; i < (int)s.size(); i++) {
        if (s[i] == '(') {
            st.push(i);
        } else {
            st.pop();
            if (st.empty()) st.push(i);
            else maxLen = max(maxLen, i - st.top());
        }
    }
    return maxLen;
}

// Minimum Remove to Make Valid Parentheses
string min_remove_valid_parens(string s) {
    stack<int> st;
    for (int i = 0; i < (int)s.size(); i++) {
        if (s[i] == '(') st.push(i);
        else if (s[i] == ')') {
            if (!st.empty() && s[st.top()] == '(') st.pop();
            else st.push(i);
        }
    }
    while (!st.empty()) { s[st.top()] = '#'; st.pop(); }
    s.erase(remove(s.begin(), s.end(), '#'), s.end());
    return s;
}

// ═══════════════════════════════════════════════════════════
//     D E C O D E / S E R I A L I Z E   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Decode String — "3[a2[c]]" → "accaccacc"
string decode_string(const string& s) {
    stack<string> str_stack;
    stack<int> num_stack;
    string cur;
    int num = 0;
    for (char c : s) {
        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == '[') {
            str_stack.push(cur);
            num_stack.push(num);
            cur = "";
            num = 0;
        } else if (c == ']') {
            string tmp = str_stack.top(); str_stack.pop();
            int rep = num_stack.top(); num_stack.pop();
            for (int i = 0; i < rep; i++) tmp += cur;
            cur = tmp;
        } else {
            cur += c;
        }
    }
    return cur;
}

// ═══════════════════════════════════════════════════════════
//     C O L L I S I O N   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Asteroid Collision — positive = right, negative = left
vector<int> asteroid_collision(const vector<int>& asteroids) {
    vector<int> st;
    for (int a : asteroids) {
        bool alive = true;
        while (alive && a < 0 && !st.empty() && st.back() > 0) {
            alive = st.back() < -a;
            if (st.back() <= -a) st.pop_back();
        }
        if (alive) st.push_back(a);
    }
    return st;
}

// ═══════════════════════════════════════════════════════════
//     P R I O R I T Y   Q U E U E   P A T T E R N S
// ═══════════════════════════════════════════════════════════

// Top K Frequent Elements — O(N log K)
vector<int> top_k_frequent(const vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int x : nums) freq[x]++;
    // Min-heap of (freq, element)
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    for (auto& [num, cnt] : freq) {
        pq.push({cnt, num});
        if ((int)pq.size() > k) pq.pop();
    }
    vector<int> result;
    while (!pq.empty()) { result.push_back(pq.top().second); pq.pop(); }
    return result;
}

// Bucket sort approach — O(N)
vector<int> top_k_frequent_bucket(const vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int x : nums) freq[x]++;
    vector<vector<int>> buckets(nums.size() + 1);
    for (auto& [num, cnt] : freq) buckets[cnt].push_back(num);
    vector<int> result;
    for (int i = buckets.size() - 1; i >= 0 && (int)result.size() < k; i--)
        for (int x : buckets[i]) { result.push_back(x); if ((int)result.size() == k) break; }
    return result;
}

// K Closest Points to Origin — O(N log K)
vector<vector<int>> k_closest(vector<vector<int>>& points, int k) {
    // Max-heap by distance
    auto cmp = [](const vector<int>& a, const vector<int>& b) {
        return a[0]*a[0]+a[1]*a[1] < b[0]*b[0]+b[1]*b[1]; };
    priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> pq(cmp);
    for (auto& p : points) {
        pq.push(p);
        if ((int)pq.size() > k) pq.pop();
    }
    vector<vector<int>> res;
    while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
    return res;
}

// Smallest Range Covering Elements from K Lists
vector<int> smallest_range(const vector<vector<int>>& nums) {
    // min-heap: (value, list_index, element_index)
    auto cmp = [](tuple<int,int,int>& a, tuple<int,int,int>& b) {
        return get<0>(a) > get<0>(b); };
    priority_queue<tuple<int,int,int>, vector<tuple<int,int,int>>, decltype(cmp)> pq(cmp);
    int curMax = INT_MIN;
    for (int i = 0; i < (int)nums.size(); i++) {
        pq.push({nums[i][0], i, 0});
        curMax = max(curMax, nums[i][0]);
    }
    int bestL = 0, bestR = INT_MAX;
    while (true) {
        auto [val, li, ei] = pq.top(); pq.pop();
        if (curMax - val < bestR - bestL) { bestL = val; bestR = curMax; }
        if (ei + 1 == (int)nums[li].size()) break;
        pq.push({nums[li][ei + 1], li, ei + 1});
        curMax = max(curMax, nums[li][ei + 1]);
    }
    return {bestL, bestR};
}

// Median from Data Stream (two heaps)
struct MedianFinder {
    priority_queue<int> lo;                            // max-heap (lower half)
    priority_queue<int, vector<int>, greater<int>> hi; // min-heap (upper half)

    void addNum(int num) {
        lo.push(num);
        hi.push(lo.top()); lo.pop();
        if (hi.size() > lo.size()) { lo.push(hi.top()); hi.pop(); }
    }

    double findMedian() {
        return lo.size() > hi.size() ? lo.top() : (lo.top() + hi.top()) / 2.0;
    }
};

// Merge K Sorted Lists — O(N log K)
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v) : val(v), next(nullptr) {}
};

ListNode* merge_k_lists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    for (auto* l : lists) if (l) pq.push(l);
    auto dummy = new ListNode(0);
    auto tail = dummy;
    while (!pq.empty()) {
        auto* node = pq.top(); pq.pop();
        tail->next = node;
        tail = tail->next;
        if (node->next) pq.push(node->next);
    }
    return dummy->next;
}

// ═══════════════════════════════════════════════════════════
//     S L I D I N G   W I N D O W   M A X I M U M
// ═══════════════════════════════════════════════════════════

// Sliding Window Maximum — O(N) with deque
vector<int> max_sliding_window(const vector<int>& nums, int k) {
    deque<int> dq; // indices of decreasing elements
    vector<int> result;
    for (int i = 0; i < (int)nums.size(); i++) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(nums[dq.front()]);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
//     M O N O T O N I C   Q U E U E   (template)
// ═══════════════════════════════════════════════════════════

// Generic monotonic queue — maintains max (or min with custom comparator)
template<typename T, typename Comp = less<T>>
struct MonotonicQueue {
    deque<pair<T, int>> dq; // (value, index)
    Comp comp;

    void push(T val, int idx) {
        while (!dq.empty() && !comp(dq.back().first, val)) dq.pop_back();
        dq.push_back({val, idx});
    }

    void pop_expired(int min_idx) {
        while (!dq.empty() && dq.front().second < min_idx) dq.pop_front();
    }

    T front() { return dq.front().first; }
    bool empty() { return dq.empty(); }
};

// ═══════════════════════════════════════════════════════════
//     S T A C K - B A S E D   D E S I G N
// ═══════════════════════════════════════════════════════════

// Min Stack — O(1) push, pop, top, getMin
struct MinStack {
    stack<pair<int,int>> st; // (value, current_min)

    void push(int val) {
        int mn = st.empty() ? val : min(val, st.top().second);
        st.push({val, mn});
    }
    void pop() { st.pop(); }
    int top() { return st.top().first; }
    int getMin() { return st.top().second; }
};

// Max Stack — O(log N) push, pop, top, peekMax, popMax
struct MaxStack {
    stack<pair<int,int>> st; // (value, id)
    map<int, vector<int>> val_to_ids;  // value -> list of ids (sorted)
    set<int> removed; // ids that have been "soft deleted"
    int id = 0;

    void push(int x) {
        st.push({x, id});
        val_to_ids[x].push_back(id);
        id++;
    }

    int top() {
        cleanup_top();
        return st.top().first;
    }

    int pop() {
        cleanup_top();
        auto [val, vid] = st.top(); st.pop();
        removed.insert(vid);
        val_to_ids[val].pop_back();
        if (val_to_ids[val].empty()) val_to_ids.erase(val);
        return val;
    }

    int peekMax() {
        return val_to_ids.rbegin()->first;
    }

    int popMax() {
        int mx = val_to_ids.rbegin()->first;
        int vid = val_to_ids[mx].back();
        val_to_ids[mx].pop_back();
        if (val_to_ids[mx].empty()) val_to_ids.erase(mx);
        removed.insert(vid);
        return mx;
    }

private:
    void cleanup_top() {
        while (!st.empty() && removed.count(st.top().second)) {
            removed.erase(st.top().second);
            st.pop();
        }
    }
};
