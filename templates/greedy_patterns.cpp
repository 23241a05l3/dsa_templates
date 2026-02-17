/*
 * ============================================================
 *     GREEDY PATTERNS — FAANG INTERVIEW TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Activity Selection (max non-overlapping intervals)
 *    2.  Fractional Knapsack
 *    3.  Job Sequencing with Deadlines
 *    4.  Huffman Coding
 *    5.  Partition Labels
 *    6.  Gas Station (circular tour)
 *    7.  Candy Distribution
 *    8.  Queue Reconstruction by Height
 *    9.  Min Arrows to Burst Balloons
 *   10.  Remove K Digits for Smallest Number
 *   11.  Task Scheduler
 *   12.  Reorganize String
 *   13.  Boats to Save People
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
// 1. ACTIVITY SELECTION — Max Non-Overlapping Intervals
// ═══════════════════════════════════════════════════════════
int activity_selection(vector<pair<int,int>>& intervals) {
    // {start, end} — sort by end time
    sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
        return a.second < b.second;
    });
    int count = 0, last_end = INT_MIN;
    for (auto& [s, e] : intervals) {
        if (s >= last_end) { count++; last_end = e; }
    }
    return count;
}

// Equivalent: min intervals to REMOVE for no overlap
int erase_overlap_intervals(vector<pair<int,int>>& intervals) {
    return (int)intervals.size() - activity_selection(intervals);
}

// ═══════════════════════════════════════════════════════════
// 2. FRACTIONAL KNAPSACK — O(N log N)
// ═══════════════════════════════════════════════════════════
double fractional_knapsack(int W, vector<pair<int,int>>& items) {
    // items = {value, weight}
    // Sort by value/weight ratio descending
    sort(items.begin(), items.end(), [](auto& a, auto& b) {
        return (double)a.first / a.second > (double)b.first / b.second;
    });
    double total = 0;
    for (auto& [v, w] : items) {
        if (W >= w) { total += v; W -= w; }
        else { total += (double)v * W / w; break; }
    }
    return total;
}

// ═══════════════════════════════════════════════════════════
// 3. JOB SEQUENCING WITH DEADLINES — O(N log N)
// ═══════════════════════════════════════════════════════════
// Each job has deadline and profit. One job per time slot. Maximize profit.
int job_sequencing(vector<tuple<int,int,int>>& jobs) {
    // jobs = {id, deadline, profit}
    sort(jobs.begin(), jobs.end(), [](auto& a, auto& b) {
        return get<2>(a) > get<2>(b); // sort by profit desc
    });
    int max_deadline = 0;
    for (auto& [id, d, p] : jobs) max_deadline = max(max_deadline, d);

    // DSU to find latest available slot ≤ deadline
    vector<int> parent(max_deadline + 1);
    iota(parent.begin(), parent.end(), 0);
    function<int(int)> find = [&](int x) -> int {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };

    int total_profit = 0;
    for (auto& [id, d, p] : jobs) {
        int slot = find(d);
        if (slot > 0) {
            total_profit += p;
            parent[slot] = slot - 1; // mark slot as used
        }
    }
    return total_profit;
}

// ═══════════════════════════════════════════════════════════
// 4. HUFFMAN CODING — O(N log N)
// ═══════════════════════════════════════════════════════════
struct HuffmanNode {
    int freq;
    char ch;
    HuffmanNode *left, *right;
    HuffmanNode(char c, int f) : freq(f), ch(c), left(nullptr), right(nullptr) {}
};

HuffmanNode* build_huffman(const unordered_map<char, int>& freq) {
    auto cmp = [](HuffmanNode* a, HuffmanNode* b) { return a->freq > b->freq; };
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, decltype(cmp)> pq(cmp);
    for (auto& [c, f] : freq) pq.push(new HuffmanNode(c, f));

    while (pq.size() > 1) {
        auto left = pq.top(); pq.pop();
        auto right = pq.top(); pq.pop();
        auto parent = new HuffmanNode('\0', left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }
    return pq.top();
}

void build_codes(HuffmanNode* root, string code, unordered_map<char, string>& codes) {
    if (!root) return;
    if (root->ch != '\0') codes[root->ch] = code;
    build_codes(root->left, code + "0", codes);
    build_codes(root->right, code + "1", codes);
}

// ═══════════════════════════════════════════════════════════
// 5. PARTITION LABELS
// ═══════════════════════════════════════════════════════════
// Split string into max parts where each char appears in at most one part
vector<int> partition_labels(const string& s) {
    vector<int> last(26, 0);
    for (int i = 0; i < (int)s.size(); i++) last[s[i] - 'a'] = i;

    vector<int> result;
    int start = 0, end = 0;
    for (int i = 0; i < (int)s.size(); i++) {
        end = max(end, last[s[i] - 'a']);
        if (i == end) {
            result.push_back(end - start + 1);
            start = end + 1;
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
// 6. GAS STATION — Circular Tour
// ═══════════════════════════════════════════════════════════
// Find starting station to complete circuit (return -1 if impossible)
int gas_station(const vector<int>& gas, const vector<int>& cost) {
    int total = 0, tank = 0, start = 0;
    for (int i = 0; i < (int)gas.size(); i++) {
        int diff = gas[i] - cost[i];
        total += diff;
        tank += diff;
        if (tank < 0) { start = i + 1; tank = 0; }
    }
    return total >= 0 ? start : -1;
}

// ═══════════════════════════════════════════════════════════
// 7. CANDY DISTRIBUTION — Min candies to satisfy neighbor constraints
// ═══════════════════════════════════════════════════════════
int candy(const vector<int>& ratings) {
    int n = ratings.size();
    vector<int> candies(n, 1);
    // Left pass: if rating[i] > rating[i-1], give more than left neighbor
    for (int i = 1; i < n; i++)
        if (ratings[i] > ratings[i-1]) candies[i] = candies[i-1] + 1;
    // Right pass: if rating[i] > rating[i+1], must be more than right neighbor
    for (int i = n-2; i >= 0; i--)
        if (ratings[i] > ratings[i+1]) candies[i] = max(candies[i], candies[i+1] + 1);
    return accumulate(candies.begin(), candies.end(), 0);
}

// ═══════════════════════════════════════════════════════════
// 8. QUEUE RECONSTRUCTION BY HEIGHT
// ═══════════════════════════════════════════════════════════
// People = {height, k} where k = number of people taller in front
vector<pair<int,int>> reconstruct_queue(vector<pair<int,int>>& people) {
    // Sort: tallest first, ties by k ascending
    sort(people.begin(), people.end(), [](auto& a, auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    vector<pair<int,int>> result;
    for (auto& p : people)
        result.insert(result.begin() + p.second, p);
    return result;
}

// ═══════════════════════════════════════════════════════════
// 9. MIN ARROWS TO BURST BALLOONS (= Non-overlapping intervals)
// ═══════════════════════════════════════════════════════════
int min_arrows(vector<pair<int,int>>& points) {
    if (points.empty()) return 0;
    sort(points.begin(), points.end(), [](auto& a, auto& b) {
        return a.second < b.second;
    });
    int arrows = 1, end = points[0].second;
    for (int i = 1; i < (int)points.size(); i++) {
        if (points[i].first > end) { arrows++; end = points[i].second; }
    }
    return arrows;
}

// ═══════════════════════════════════════════════════════════
// 10. REMOVE K DIGITS — make smallest number
// ═══════════════════════════════════════════════════════════
string remove_k_digits(const string& num, int k) {
    string stk;
    for (char c : num) {
        while (k > 0 && !stk.empty() && stk.back() > c) {
            stk.pop_back();
            k--;
        }
        stk.push_back(c);
    }
    while (k > 0) { stk.pop_back(); k--; }
    // Remove leading zeros
    int start = 0;
    while (start < (int)stk.size() && stk[start] == '0') start++;
    string result = stk.substr(start);
    return result.empty() ? "0" : result;
}

// ═══════════════════════════════════════════════════════════
// 11. TASK SCHEDULER — Min intervals to schedule all tasks with cooldown n
// ═══════════════════════════════════════════════════════════
int least_interval(const vector<char>& tasks, int n) {
    vector<int> freq(26, 0);
    for (char t : tasks) freq[t - 'A']++;
    int max_freq = *max_element(freq.begin(), freq.end());
    int max_count = count(freq.begin(), freq.end(), max_freq);
    // (max_freq - 1) full chunks of size (n+1) + last partial chunk of max_count
    int result = (max_freq - 1) * (n + 1) + max_count;
    return max(result, (int)tasks.size());
}

// ═══════════════════════════════════════════════════════════
// 12. REORGANIZE STRING — no two adjacent chars same
// ═══════════════════════════════════════════════════════════
string reorganize_string(const string& s) {
    vector<int> freq(26, 0);
    for (char c : s) freq[c - 'a']++;
    // Check feasibility
    int max_freq = *max_element(freq.begin(), freq.end());
    if (max_freq > ((int)s.size() + 1) / 2) return "";

    // Max-heap of {freq, char}
    priority_queue<pair<int,char>> pq;
    for (int i = 0; i < 26; i++)
        if (freq[i]) pq.push({freq[i], 'a' + i});

    string result;
    while (pq.size() >= 2) {
        auto [f1, c1] = pq.top(); pq.pop();
        auto [f2, c2] = pq.top(); pq.pop();
        result += c1;
        result += c2;
        if (f1 > 1) pq.push({f1 - 1, c1});
        if (f2 > 1) pq.push({f2 - 1, c2});
    }
    if (!pq.empty()) result += pq.top().second;
    return result;
}

// ═══════════════════════════════════════════════════════════
// 13. BOATS TO SAVE PEOPLE — pair lightest with heaviest
// ═══════════════════════════════════════════════════════════
int num_boats(vector<int>& people, int limit) {
    sort(people.begin(), people.end());
    int l = 0, r = (int)people.size() - 1, boats = 0;
    while (l <= r) {
        if (people[l] + people[r] <= limit) l++;
        r--;
        boats++;
    }
    return boats;
}
