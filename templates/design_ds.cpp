/*
 * ============================================================
 *     DESIGN DATA STRUCTURES — FAANG INTERVIEW TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  LFU Cache — O(1) get/put
 *    2.  Randomized Set — O(1) insert/delete/getRandom
 *    3.  All O(1) Data Structure (Inc/Dec/GetMax/GetMin)
 *    4.  Trie with Wildcard Search (.)
 *    5.  Skip List
 *    6.  BST Iterator — O(1) amortized next()
 *    7.  Flatten Nested List Iterator
 *    8.  Design Twitter (Merge K Sorted Feeds)
 *    9.  Time-Based Key-Value Store
 *   10.  Serialize / Deserialize N-ary Tree
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ═══════════════════════════════════════════════════════════
// 1. LFU CACHE — O(1) get / put
// ═══════════════════════════════════════════════════════════
struct LFUCache {
    int cap, min_freq;
    unordered_map<int, pair<int, int>> key_val_freq;       // key -> {value, freq}
    unordered_map<int, list<int>> freq_list;               // freq -> ordered list of keys
    unordered_map<int, list<int>::iterator> key_iter;      // key -> iterator in freq_list

    LFUCache(int capacity) : cap(capacity), min_freq(0) {}

    int get(int key) {
        if (!key_val_freq.count(key)) return -1;
        touch(key);
        return key_val_freq[key].first;
    }

    void put(int key, int value) {
        if (cap <= 0) return;
        if (key_val_freq.count(key)) {
            key_val_freq[key].first = value;
            touch(key);
            return;
        }
        if ((int)key_val_freq.size() >= cap) {
            // Evict LFU (ties broken by LRU = front of min_freq list)
            int evict = freq_list[min_freq].front();
            freq_list[min_freq].pop_front();
            if (freq_list[min_freq].empty()) freq_list.erase(min_freq);
            key_val_freq.erase(evict);
            key_iter.erase(evict);
        }
        key_val_freq[key] = {value, 1};
        freq_list[1].push_back(key);
        key_iter[key] = prev(freq_list[1].end());
        min_freq = 1;
    }

private:
    void touch(int key) {
        int freq = key_val_freq[key].second;
        key_val_freq[key].second++;
        freq_list[freq].erase(key_iter[key]);
        if (freq_list[freq].empty()) {
            freq_list.erase(freq);
            if (min_freq == freq) min_freq++;
        }
        freq_list[freq + 1].push_back(key);
        key_iter[key] = prev(freq_list[freq + 1].end());
    }
};

// ═══════════════════════════════════════════════════════════
// 2. RANDOMIZED SET — O(1) insert / delete / getRandom
// ═══════════════════════════════════════════════════════════
struct RandomizedSet {
    vector<int> nums;
    unordered_map<int, int> idx; // val -> index in nums
    mt19937 rng{(unsigned)chrono::steady_clock::now().time_since_epoch().count()};

    bool insert(int val) {
        if (idx.count(val)) return false;
        idx[val] = nums.size();
        nums.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!idx.count(val)) return false;
        int i = idx[val];
        int last = nums.back();
        nums[i] = last;
        idx[last] = i;
        nums.pop_back();
        idx.erase(val);
        return true;
    }

    int getRandom() {
        return nums[rng() % nums.size()];
    }
};

// ═══════════════════════════════════════════════════════════
// 3. ALL O(1) DATA STRUCTURE — Inc/Dec/GetMaxKey/GetMinKey
// ═══════════════════════════════════════════════════════════
struct AllOne {
    struct Bucket {
        int count;
        unordered_set<string> keys;
    };
    list<Bucket> buckets;
    unordered_map<string, list<Bucket>::iterator> key_bucket;

    void inc(const string& key) {
        if (!key_bucket.count(key)) {
            // New key with count 1
            if (buckets.empty() || buckets.front().count != 1)
                buckets.push_front({1, {}});
            buckets.front().keys.insert(key);
            key_bucket[key] = buckets.begin();
        } else {
            auto cur = key_bucket[key];
            int new_count = cur->count + 1;
            auto next_it = next(cur);
            if (next_it == buckets.end() || next_it->count != new_count)
                next_it = buckets.insert(next_it, {new_count, {}});
            next_it->keys.insert(key);
            cur->keys.erase(key);
            if (cur->keys.empty()) buckets.erase(cur);
            key_bucket[key] = next_it;
        }
    }

    void dec(const string& key) {
        if (!key_bucket.count(key)) return;
        auto cur = key_bucket[key];
        int new_count = cur->count - 1;
        if (new_count == 0) {
            cur->keys.erase(key);
            if (cur->keys.empty()) buckets.erase(cur);
            key_bucket.erase(key);
        } else {
            auto prev_it = cur;
            if (cur == buckets.begin() || prev(cur)->count != new_count) {
                prev_it = buckets.insert(cur, {new_count, {}});
            } else {
                prev_it = prev(cur);
            }
            prev_it->keys.insert(key);
            cur->keys.erase(key);
            if (cur->keys.empty()) buckets.erase(cur);
            key_bucket[key] = prev_it;
        }
    }

    string getMaxKey() {
        return buckets.empty() ? "" : *buckets.back().keys.begin();
    }

    string getMinKey() {
        return buckets.empty() ? "" : *buckets.front().keys.begin();
    }
};

// ═══════════════════════════════════════════════════════════
// 4. TRIE WITH WILDCARD SEARCH (. matches any char)
// ═══════════════════════════════════════════════════════════
struct WildcardTrie {
    struct Node {
        Node* ch[26] = {};
        bool is_end = false;
    };
    Node* root = new Node();

    void addWord(const string& word) {
        Node* cur = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!cur->ch[idx]) cur->ch[idx] = new Node();
            cur = cur->ch[idx];
        }
        cur->is_end = true;
    }

    bool search(const string& word) {
        return dfs(word, 0, root);
    }

private:
    bool dfs(const string& word, int i, Node* node) {
        if (!node) return false;
        if (i == (int)word.size()) return node->is_end;
        if (word[i] == '.') {
            for (int c = 0; c < 26; c++)
                if (dfs(word, i + 1, node->ch[c])) return true;
            return false;
        }
        return dfs(word, i + 1, node->ch[word[i] - 'a']);
    }
};

// ═══════════════════════════════════════════════════════════
// 5. SKIP LIST — O(log N) expected search/insert/delete
// ═══════════════════════════════════════════════════════════
struct SkipList {
    static const int MAX_LEVEL = 16;
    struct Node {
        int val;
        vector<Node*> next;
        Node(int v, int level) : val(v), next(level + 1, nullptr) {}
    };
    Node* head = new Node(INT_MIN, MAX_LEVEL);
    int level = 0;
    mt19937 rng{42};

    int randomLevel() {
        int lvl = 0;
        while (lvl < MAX_LEVEL && (rng() & 1)) lvl++;
        return lvl;
    }

    bool search(int target) {
        Node* cur = head;
        for (int i = level; i >= 0; i--)
            while (cur->next[i] && cur->next[i]->val < target)
                cur = cur->next[i];
        cur = cur->next[0];
        return cur && cur->val == target;
    }

    void add(int num) {
        vector<Node*> update(MAX_LEVEL + 1, head);
        Node* cur = head;
        for (int i = level; i >= 0; i--) {
            while (cur->next[i] && cur->next[i]->val < num)
                cur = cur->next[i];
            update[i] = cur;
        }
        int new_level = randomLevel();
        level = max(level, new_level);
        Node* node = new Node(num, new_level);
        for (int i = 0; i <= new_level; i++) {
            node->next[i] = update[i]->next[i];
            update[i]->next[i] = node;
        }
    }

    bool erase(int num) {
        vector<Node*> update(MAX_LEVEL + 1, nullptr);
        Node* cur = head;
        for (int i = level; i >= 0; i--) {
            while (cur->next[i] && cur->next[i]->val < num)
                cur = cur->next[i];
            update[i] = cur;
        }
        cur = cur->next[0];
        if (!cur || cur->val != num) return false;
        for (int i = 0; i <= level; i++) {
            if (update[i]->next[i] != cur) break;
            update[i]->next[i] = cur->next[i];
        }
        delete cur;
        while (level > 0 && !head->next[level]) level--;
        return true;
    }
};

// ═══════════════════════════════════════════════════════════
// 6. BST ITERATOR — O(1) amortized next(), O(h) space
// ═══════════════════════════════════════════════════════════
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

struct BSTIterator {
    stack<TreeNode*> stk;

    BSTIterator(TreeNode* root) { pushLeft(root); }

    int next() {
        TreeNode* node = stk.top(); stk.pop();
        pushLeft(node->right);
        return node->val;
    }

    bool hasNext() { return !stk.empty(); }

private:
    void pushLeft(TreeNode* node) {
        while (node) { stk.push(node); node = node->left; }
    }
};

// ═══════════════════════════════════════════════════════════
// 7. FLATTEN NESTED LIST ITERATOR
// ═══════════════════════════════════════════════════════════
// Simulating NestedInteger interface for template purposes
struct NestedInteger {
    bool is_int;
    int val;
    vector<NestedInteger> list;
    NestedInteger(int v) : is_int(true), val(v) {}
    NestedInteger(vector<NestedInteger> l) : is_int(false), val(0), list(l) {}
    bool isInteger() const { return is_int; }
    int getInteger() const { return val; }
    const vector<NestedInteger>& getList() const { return list; }
};

struct NestedIterator {
    stack<pair<const vector<NestedInteger>*, int>> stk;

    NestedIterator(const vector<NestedInteger>& nestedList) {
        stk.push({&nestedList, 0});
        advance();
    }

    int next() {
        auto& [list, idx] = stk.top();
        int val = (*list)[idx].getInteger();
        idx++;
        advance();
        return val;
    }

    bool hasNext() { return !stk.empty(); }

private:
    void advance() {
        while (!stk.empty()) {
            auto& [list, idx] = stk.top();
            if (idx >= (int)list->size()) { stk.pop(); continue; }
            if ((*list)[idx].isInteger()) return; // ready
            auto& nested = (*list)[idx].getList();
            idx++;
            stk.push({&nested, 0});
        }
    }
};

// ═══════════════════════════════════════════════════════════
// 8. DESIGN TWITTER (Merge K Sorted Feeds)
// ═══════════════════════════════════════════════════════════
struct Twitter {
    int timestamp = 0;
    unordered_map<int, vector<pair<int, int>>> tweets; // userId -> [{time, tweetId}]
    unordered_map<int, unordered_set<int>> following;  // userId -> set of followees

    void postTweet(int userId, int tweetId) {
        tweets[userId].push_back({timestamp++, tweetId});
    }

    vector<int> getNewsFeed(int userId) {
        // Merge K sorted lists (own + followed), get top 10
        auto cmp = [](auto& a, auto& b) { return a.first < b.first; }; // max heap
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>> pq; // {time, tweetId, unused}

        auto addTweets = [&](int uid) {
            auto& tw = tweets[uid];
            for (int i = max(0, (int)tw.size() - 10); i < (int)tw.size(); i++)
                pq.push({tw[i].first, tw[i].second, 0});
        };

        addTweets(userId);
        for (int fid : following[userId]) addTweets(fid);

        vector<int> feed;
        while (!pq.empty() && (int)feed.size() < 10) {
            auto [t, tid, _] = pq.top(); pq.pop();
            feed.push_back(tid);
        }
        return feed;
    }

    void follow(int followerId, int followeeId) {
        if (followerId != followeeId) following[followerId].insert(followeeId);
    }

    void unfollow(int followerId, int followeeId) {
        following[followerId].erase(followeeId);
    }
};

// ═══════════════════════════════════════════════════════════
// 9. TIME-BASED KEY-VALUE STORE
// ═══════════════════════════════════════════════════════════
struct TimeMap {
    unordered_map<string, vector<pair<int, string>>> store; // key -> [{timestamp, value}]

    void set(const string& key, const string& value, int timestamp) {
        store[key].push_back({timestamp, value});
    }

    string get(const string& key, int timestamp) {
        if (!store.count(key)) return "";
        auto& v = store[key];
        // Binary search for largest timestamp <= given timestamp
        int lo = 0, hi = (int)v.size() - 1, ans = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (v[mid].first <= timestamp) { ans = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return ans >= 0 ? v[ans].second : "";
    }
};

// ═══════════════════════════════════════════════════════════
// 10. SERIALIZE / DESERIALIZE N-ARY TREE
// ═══════════════════════════════════════════════════════════
struct NaryNode {
    int val;
    vector<NaryNode*> children;
    NaryNode(int v) : val(v) {}
};

// Encoding: val [child_count] children...
string serialize_nary(NaryNode* root) {
    if (!root) return "#";
    string res = to_string(root->val) + " " + to_string(root->children.size());
    for (auto* child : root->children)
        res += " " + serialize_nary(child);
    return res;
}

NaryNode* deserialize_nary_helper(istringstream& ss) {
    string token;
    if (!(ss >> token) || token == "#") return nullptr;
    auto* node = new NaryNode(stoi(token));
    int count;
    ss >> count;
    for (int i = 0; i < count; i++)
        node->children.push_back(deserialize_nary_helper(ss));
    return node;
}
NaryNode* deserialize_nary(const string& data) {
    istringstream ss(data);
    return deserialize_nary_helper(ss);
}
