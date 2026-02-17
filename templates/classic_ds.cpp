/*
 * ============================================================
 *       CLASSIC DATA STRUCTURES — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Binary Search Tree (BST) — full implementation
 *    2.  AVL Tree (Self-balancing BST)
 *    3.  Min Heap / Max Heap (manual array-based)
 *    4.  Singly Linked List (full: reverse, cycle, merge, etc.)
 *    5.  Doubly Linked List (full implementation)
 *    6.  Stack (array-based + linked-list)
 *    7.  Queue / Circular Queue / Deque
 *    8.  Hash Map (chaining + open addressing)
 *    9.  LRU Cache
 *   10.  Interval Tree
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ═══════════════════════════════════════════════════════════
// 1. BINARY SEARCH TREE — Full Implementation
// ═══════════════════════════════════════════════════════════
struct BST {
    struct Node {
        int key;
        Node *left, *right;
        Node(int k) : key(k), left(nullptr), right(nullptr) {}
    };
    Node* root = nullptr;

    // ---- Insert ----
    Node* insert(Node* node, int key) {
        if (!node) return new Node(key);
        if (key < node->key) node->left = insert(node->left, key);
        else if (key > node->key) node->right = insert(node->right, key);
        return node;
    }
    void insert(int key) { root = insert(root, key); }

    // ---- Search ----
    bool search(Node* node, int key) {
        if (!node) return false;
        if (key == node->key) return true;
        return key < node->key ? search(node->left, key) : search(node->right, key);
    }
    bool search(int key) { return search(root, key); }

    // ---- Find Min / Max ----
    Node* find_min(Node* node) { while (node->left) node = node->left; return node; }
    Node* find_max(Node* node) { while (node->right) node = node->right; return node; }
    int min_val() { return find_min(root)->key; }
    int max_val() { return find_max(root)->key; }

    // ---- Delete ----
    Node* erase(Node* node, int key) {
        if (!node) return nullptr;
        if (key < node->key) node->left = erase(node->left, key);
        else if (key > node->key) node->right = erase(node->right, key);
        else {
            if (!node->left) { Node* tmp = node->right; delete node; return tmp; }
            if (!node->right) { Node* tmp = node->left; delete node; return tmp; }
            Node* succ = find_min(node->right);
            node->key = succ->key;
            node->right = erase(node->right, succ->key);
        }
        return node;
    }
    void erase(int key) { root = erase(root, key); }

    // ---- Traversals ----
    void inorder(Node* node, vector<int>& res) {
        if (!node) return;
        inorder(node->left, res);
        res.push_back(node->key);
        inorder(node->right, res);
    }
    vector<int> inorder() { vector<int> res; inorder(root, res); return res; }

    void preorder(Node* node, vector<int>& res) {
        if (!node) return;
        res.push_back(node->key);
        preorder(node->left, res);
        preorder(node->right, res);
    }
    vector<int> preorder() { vector<int> res; preorder(root, res); return res; }

    void postorder(Node* node, vector<int>& res) {
        if (!node) return;
        postorder(node->left, res);
        postorder(node->right, res);
        res.push_back(node->key);
    }
    vector<int> postorder() { vector<int> res; postorder(root, res); return res; }

    // ---- Level Order (BFS) ----
    vector<vector<int>> level_order() {
        vector<vector<int>> res;
        if (!root) return res;
        queue<Node*> q;
        q.push(root);
        while (!q.empty()) {
            int sz = q.size();
            vector<int> level;
            while (sz--) {
                Node* cur = q.front(); q.pop();
                level.push_back(cur->key);
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res.push_back(level);
        }
        return res;
    }

    // ---- Kth Smallest (augment with subtree size for O(log N)) ----
    int kth_smallest(Node* node, int& k) {
        if (!node) return -1;
        int left = kth_smallest(node->left, k);
        if (left != -1) return left;
        if (--k == 0) return node->key;
        return kth_smallest(node->right, k);
    }
    int kth_smallest(int k) { return kth_smallest(root, k); }

    // ---- LCA in BST ----
    int lca(int a, int b) {
        Node* cur = root;
        while (cur) {
            if (a < cur->key && b < cur->key) cur = cur->left;
            else if (a > cur->key && b > cur->key) cur = cur->right;
            else return cur->key;
        }
        return -1;
    }

    // ---- Height ----
    int height(Node* node) {
        if (!node) return -1;
        return 1 + max(height(node->left), height(node->right));
    }
    int height() { return height(root); }

    // ---- Validate BST ----
    bool is_valid(Node* node, long long lo, long long hi) {
        if (!node) return true;
        if (node->key <= lo || node->key >= hi) return false;
        return is_valid(node->left, lo, node->key) && is_valid(node->right, node->key, hi);
    }
    bool is_valid() { return is_valid(root, LLONG_MIN, LLONG_MAX); }

    // ---- Floor & Ceil ----
    int floor(int key) {
        int res = INT_MIN;
        Node* cur = root;
        while (cur) {
            if (cur->key == key) return key;
            if (cur->key < key) { res = cur->key; cur = cur->right; }
            else cur = cur->left;
        }
        return res;
    }
    int ceil(int key) {
        int res = INT_MAX;
        Node* cur = root;
        while (cur) {
            if (cur->key == key) return key;
            if (cur->key > key) { res = cur->key; cur = cur->left; }
            else cur = cur->right;
        }
        return res;
    }

    // ---- Predecessor & Successor ----
    int predecessor(int key) { return floor(key - 1); }  // assuming integer keys
    int successor(int key) { return ceil(key + 1); }
};

// ═══════════════════════════════════════════════════════════
// 2. AVL TREE (Self-Balancing BST) — O(log N) all operations
// ═══════════════════════════════════════════════════════════
struct AVLTree {
    struct Node {
        int key, height;
        Node *left, *right;
        Node(int k) : key(k), height(1), left(nullptr), right(nullptr) {}
    };
    Node* root = nullptr;

    int h(Node* n) { return n ? n->height : 0; }
    int bf(Node* n) { return n ? h(n->left) - h(n->right) : 0; }
    void update_height(Node* n) { if (n) n->height = 1 + max(h(n->left), h(n->right)); }

    // Right rotation
    Node* rotate_right(Node* y) {
        Node* x = y->left;
        Node* T2 = x->right;
        x->right = y;
        y->left = T2;
        update_height(y);
        update_height(x);
        return x;
    }

    // Left rotation
    Node* rotate_left(Node* x) {
        Node* y = x->right;
        Node* T2 = y->left;
        y->left = x;
        x->right = T2;
        update_height(x);
        update_height(y);
        return y;
    }

    Node* balance(Node* node) {
        update_height(node);
        int b = bf(node);
        if (b > 1) { // Left heavy
            if (bf(node->left) < 0) node->left = rotate_left(node->left); // LR case
            return rotate_right(node); // LL case
        }
        if (b < -1) { // Right heavy
            if (bf(node->right) > 0) node->right = rotate_right(node->right); // RL case
            return rotate_left(node); // RR case
        }
        return node;
    }

    Node* insert(Node* node, int key) {
        if (!node) return new Node(key);
        if (key < node->key) node->left = insert(node->left, key);
        else if (key > node->key) node->right = insert(node->right, key);
        else return node; // duplicate
        return balance(node);
    }
    void insert(int key) { root = insert(root, key); }

    Node* find_min(Node* node) { while (node->left) node = node->left; return node; }

    Node* erase(Node* node, int key) {
        if (!node) return nullptr;
        if (key < node->key) node->left = erase(node->left, key);
        else if (key > node->key) node->right = erase(node->right, key);
        else {
            if (!node->left || !node->right) {
                Node* tmp = node->left ? node->left : node->right;
                delete node;
                return tmp;
            }
            Node* succ = find_min(node->right);
            node->key = succ->key;
            node->right = erase(node->right, succ->key);
        }
        return balance(node);
    }
    void erase(int key) { root = erase(root, key); }

    bool search(Node* node, int key) {
        if (!node) return false;
        if (key == node->key) return true;
        return key < node->key ? search(node->left, key) : search(node->right, key);
    }
    bool search(int key) { return search(root, key); }

    void inorder(Node* node, vector<int>& res) {
        if (!node) return;
        inorder(node->left, res);
        res.push_back(node->key);
        inorder(node->right, res);
    }
    vector<int> inorder() { vector<int> res; inorder(root, res); return res; }
};

// ═══════════════════════════════════════════════════════════
// 3. MIN HEAP / MAX HEAP — Manual Array-Based Implementation
// ═══════════════════════════════════════════════════════════

// Min Heap
struct MinHeap {
    vector<int> heap;

    int parent(int i) { return (i - 1) / 2; }
    int left(int i) { return 2 * i + 1; }
    int right(int i) { return 2 * i + 2; }

    void sift_up(int i) {
        while (i > 0 && heap[parent(i)] > heap[i]) {
            swap(heap[parent(i)], heap[i]);
            i = parent(i);
        }
    }

    void sift_down(int i) {
        int n = heap.size();
        while (true) {
            int smallest = i;
            int l = left(i), r = right(i);
            if (l < n && heap[l] < heap[smallest]) smallest = l;
            if (r < n && heap[r] < heap[smallest]) smallest = r;
            if (smallest == i) break;
            swap(heap[i], heap[smallest]);
            i = smallest;
        }
    }

    void push(int val) {
        heap.push_back(val);
        sift_up(heap.size() - 1);
    }

    int top() { return heap[0]; }

    void pop() {
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) sift_down(0);
    }

    int size() { return heap.size(); }
    bool empty() { return heap.empty(); }

    // Build heap from array — O(N)
    void build(const vector<int>& arr) {
        heap = arr;
        for (int i = (int)heap.size() / 2 - 1; i >= 0; i--)
            sift_down(i);
    }

    // Heap sort — O(N log N) in-place
    static vector<int> heap_sort(vector<int> arr) {
        MinHeap h;
        h.build(arr);
        vector<int> sorted;
        while (!h.empty()) {
            sorted.push_back(h.top());
            h.pop();
        }
        return sorted;
    }

    // Decrease key
    void decrease_key(int i, int new_val) {
        heap[i] = new_val;
        sift_up(i);
    }

    // Delete arbitrary element at index i
    void erase(int i) {
        decrease_key(i, INT_MIN);
        pop();
    }
};

// Max Heap (same structure, flip comparisons)
struct MaxHeap {
    vector<int> heap;
    int parent(int i) { return (i - 1) / 2; }
    int left(int i) { return 2 * i + 1; }
    int right(int i) { return 2 * i + 2; }

    void sift_up(int i) {
        while (i > 0 && heap[parent(i)] < heap[i]) {
            swap(heap[parent(i)], heap[i]);
            i = parent(i);
        }
    }
    void sift_down(int i) {
        int n = heap.size();
        while (true) {
            int largest = i, l = left(i), r = right(i);
            if (l < n && heap[l] > heap[largest]) largest = l;
            if (r < n && heap[r] > heap[largest]) largest = r;
            if (largest == i) break;
            swap(heap[i], heap[largest]);
            i = largest;
        }
    }
    void push(int val) { heap.push_back(val); sift_up(heap.size() - 1); }
    int top() { return heap[0]; }
    void pop() { heap[0] = heap.back(); heap.pop_back(); if (!heap.empty()) sift_down(0); }
    int size() { return heap.size(); }
    bool empty() { return heap.empty(); }
    void build(const vector<int>& arr) { heap = arr; for (int i = heap.size()/2-1; i >= 0; i--) sift_down(i); }
};

// K-th largest element using min-heap of size K — O(N log K)
int kth_largest(const vector<int>& arr, int k) {
    priority_queue<int, vector<int>, greater<int>> pq; // min-heap
    for (int x : arr) {
        pq.push(x);
        if ((int)pq.size() > k) pq.pop();
    }
    return pq.top();
}

// Merge K sorted arrays using min-heap — O(N log K)
vector<int> merge_k_sorted(const vector<vector<int>>& arrays) {
    // {value, array_idx, element_idx}
    priority_queue<tuple<int,int,int>, vector<tuple<int,int,int>>, greater<>> pq;
    for (int i = 0; i < (int)arrays.size(); i++)
        if (!arrays[i].empty()) pq.push({arrays[i][0], i, 0});
    vector<int> result;
    while (!pq.empty()) {
        auto [val, ai, ei] = pq.top(); pq.pop();
        result.push_back(val);
        if (ei + 1 < (int)arrays[ai].size())
            pq.push({arrays[ai][ei+1], ai, ei+1});
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
// 4. SINGLY LINKED LIST — Full Implementation
// ═══════════════════════════════════════════════════════════
struct SinglyLinkedList {
    struct Node {
        int val;
        Node* next;
        Node(int v, Node* n = nullptr) : val(v), next(n) {}
    };
    Node* head = nullptr;
    int sz = 0;

    // ---- Basic Operations ----
    void push_front(int val) { head = new Node(val, head); sz++; }

    void push_back(int val) {
        if (!head) { push_front(val); return; }
        Node* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = new Node(val);
        sz++;
    }

    void pop_front() {
        if (!head) return;
        Node* tmp = head;
        head = head->next;
        delete tmp;
        sz--;
    }

    void insert_at(int idx, int val) {
        if (idx == 0) { push_front(val); return; }
        Node* cur = head;
        for (int i = 0; i < idx - 1 && cur; i++) cur = cur->next;
        if (cur) { cur->next = new Node(val, cur->next); sz++; }
    }

    void erase_at(int idx) {
        if (idx == 0) { pop_front(); return; }
        Node* cur = head;
        for (int i = 0; i < idx - 1 && cur; i++) cur = cur->next;
        if (cur && cur->next) {
            Node* tmp = cur->next;
            cur->next = tmp->next;
            delete tmp;
            sz--;
        }
    }

    int get(int idx) {
        Node* cur = head;
        for (int i = 0; i < idx; i++) cur = cur->next;
        return cur->val;
    }

    // ---- Reverse ----
    void reverse() {
        Node *prev = nullptr, *cur = head, *next = nullptr;
        while (cur) {
            next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        head = prev;
    }

    // ---- Reverse range [l, r] (0-indexed) ----
    void reverse_range(int l, int r) {
        if (l >= r) return;
        Node dummy(0, head);
        Node* pre = &dummy;
        for (int i = 0; i < l; i++) pre = pre->next;
        Node* start = pre->next;
        for (int i = 0; i < r - l; i++) {
            Node* tmp = start->next;
            start->next = tmp->next;
            tmp->next = pre->next;
            pre->next = tmp;
        }
        head = dummy.next;
    }

    // ---- Detect Cycle (Floyd's) ----
    bool has_cycle() {
        Node *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }

    // ---- Find Cycle Start ----
    Node* cycle_start() {
        Node *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                slow = head;
                while (slow != fast) { slow = slow->next; fast = fast->next; }
                return slow;
            }
        }
        return nullptr;
    }

    // ---- Find Middle ----
    Node* middle() {
        Node *slow = head, *fast = head;
        while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
        return slow;
    }

    // ---- Merge Two Sorted Lists ----
    static Node* merge_sorted(Node* a, Node* b) {
        Node dummy(0);
        Node* tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) { tail->next = a; a = a->next; }
            else { tail->next = b; b = b->next; }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }

    // ---- Merge Sort on Linked List — O(N log N) ----
    static Node* merge_sort(Node* head) {
        if (!head || !head->next) return head;
        // Find middle
        Node *slow = head, *fast = head->next;
        while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
        Node* mid = slow->next;
        slow->next = nullptr;
        return merge_sorted(merge_sort(head), merge_sort(mid));
    }
    void sort() { head = merge_sort(head); }

    // ---- Remove Nth from End ----
    void remove_nth_from_end(int n) {
        Node dummy(0, head);
        Node *fast = &dummy, *slow = &dummy;
        for (int i = 0; i <= n; i++) fast = fast->next;
        while (fast) { slow = slow->next; fast = fast->next; }
        Node* tmp = slow->next;
        slow->next = tmp->next;
        delete tmp;
        head = dummy.next;
        sz--;
    }

    // ---- Check Palindrome ----
    bool is_palindrome() {
        if (!head || !head->next) return true;
        // Find middle
        Node *slow = head, *fast = head;
        while (fast->next && fast->next->next) { slow = slow->next; fast = fast->next->next; }
        // Reverse second half
        Node *prev = nullptr, *cur = slow->next;
        while (cur) { Node* next = cur->next; cur->next = prev; prev = cur; cur = next; }
        // Compare
        Node *p1 = head, *p2 = prev;
        bool result = true;
        while (p2) {
            if (p1->val != p2->val) { result = false; break; }
            p1 = p1->next; p2 = p2->next;
        }
        // Restore (optional)
        cur = prev; prev = nullptr;
        while (cur) { Node* next = cur->next; cur->next = prev; prev = cur; cur = next; }
        slow->next = prev;
        return result;
    }

    // ---- Remove Duplicates (sorted list) ----
    void remove_duplicates() {
        Node* cur = head;
        while (cur && cur->next) {
            if (cur->val == cur->next->val) {
                Node* tmp = cur->next;
                cur->next = tmp->next;
                delete tmp; sz--;
            } else cur = cur->next;
        }
    }

    // ---- Intersection Point of Two Lists ----
    static Node* intersection(Node* a, Node* b) {
        Node *pa = a, *pb = b;
        while (pa != pb) {
            pa = pa ? pa->next : b;
            pb = pb ? pb->next : a;
        }
        return pa;
    }

    // ---- To Vector ----
    vector<int> to_vector() {
        vector<int> res;
        for (Node* cur = head; cur; cur = cur->next) res.push_back(cur->val);
        return res;
    }
};

// ═══════════════════════════════════════════════════════════
// 5. DOUBLY LINKED LIST — Full Implementation
// ═══════════════════════════════════════════════════════════
struct DoublyLinkedList {
    struct Node {
        int val;
        Node *prev, *next;
        Node(int v, Node* p = nullptr, Node* n = nullptr) : val(v), prev(p), next(n) {}
    };
    Node *head = nullptr, *tail = nullptr;
    int sz = 0;

    void push_front(int val) {
        Node* node = new Node(val, nullptr, head);
        if (head) head->prev = node;
        else tail = node;
        head = node;
        sz++;
    }

    void push_back(int val) {
        Node* node = new Node(val, tail, nullptr);
        if (tail) tail->next = node;
        else head = node;
        tail = node;
        sz++;
    }

    void pop_front() {
        if (!head) return;
        Node* tmp = head;
        head = head->next;
        if (head) head->prev = nullptr;
        else tail = nullptr;
        delete tmp; sz--;
    }

    void pop_back() {
        if (!tail) return;
        Node* tmp = tail;
        tail = tail->prev;
        if (tail) tail->next = nullptr;
        else head = nullptr;
        delete tmp; sz--;
    }

    void insert_after(Node* node, int val) {
        Node* new_node = new Node(val, node, node->next);
        if (node->next) node->next->prev = new_node;
        else tail = new_node;
        node->next = new_node;
        sz++;
    }

    void erase(Node* node) {
        if (node->prev) node->prev->next = node->next;
        else head = node->next;
        if (node->next) node->next->prev = node->prev;
        else tail = node->prev;
        delete node; sz--;
    }

    void reverse() {
        Node* cur = head;
        while (cur) {
            swap(cur->prev, cur->next);
            cur = cur->prev; // was next before swap
        }
        swap(head, tail);
    }

    vector<int> to_vector() {
        vector<int> res;
        for (Node* cur = head; cur; cur = cur->next) res.push_back(cur->val);
        return res;
    }

    vector<int> to_vector_reverse() {
        vector<int> res;
        for (Node* cur = tail; cur; cur = cur->prev) res.push_back(cur->val);
        return res;
    }
};

// ═══════════════════════════════════════════════════════════
// 6. STACK — Array-based + Linked-list-based
// ═══════════════════════════════════════════════════════════

// Array-based Stack
struct ArrayStack {
    vector<int> data;
    void push(int val) { data.push_back(val); }
    int top() { return data.back(); }
    void pop() { data.pop_back(); }
    bool empty() { return data.empty(); }
    int size() { return data.size(); }
};

// Linked-list Stack
struct LLStack {
    struct Node {
        int val; Node* next;
        Node(int v, Node* n) : val(v), next(n) {}
    };
    Node* head = nullptr;
    int sz = 0;

    void push(int val) { head = new Node(val, head); sz++; }
    int top() { return head->val; }
    void pop() { Node* tmp = head; head = head->next; delete tmp; sz--; }
    bool empty() { return !head; }
    int size() { return sz; }
};

// Min Stack — O(1) getMin
struct MinStack {
    stack<int> st;
    stack<int> min_st;
    void push(int val) {
        st.push(val);
        min_st.push(min_st.empty() ? val : min(val, min_st.top()));
    }
    void pop() { st.pop(); min_st.pop(); }
    int top() { return st.top(); }
    int getMin() { return min_st.top(); }
    bool empty() { return st.empty(); }
};

// Stack using Two Queues
struct StackUsingQueues {
    queue<int> q1, q2;
    void push(int val) {
        q2.push(val);
        while (!q1.empty()) { q2.push(q1.front()); q1.pop(); }
        swap(q1, q2);
    }
    int top() { return q1.front(); }
    void pop() { q1.pop(); }
    bool empty() { return q1.empty(); }
};

// ═══════════════════════════════════════════════════════════
// 7. QUEUE / CIRCULAR QUEUE / DEQUE
// ═══════════════════════════════════════════════════════════

// Linked-list Queue
struct LLQueue {
    struct Node { int val; Node* next; Node(int v) : val(v), next(nullptr) {} };
    Node *front_node = nullptr, *back_node = nullptr;
    int sz = 0;

    void push(int val) {
        Node* node = new Node(val);
        if (back_node) back_node->next = node;
        else front_node = node;
        back_node = node; sz++;
    }
    int front() { return front_node->val; }
    void pop() {
        Node* tmp = front_node;
        front_node = front_node->next;
        if (!front_node) back_node = nullptr;
        delete tmp; sz--;
    }
    bool empty() { return !front_node; }
    int size() { return sz; }
};

// Circular Queue (Fixed Size)
struct CircularQueue {
    vector<int> data;
    int front_idx, back_idx, sz, cap;

    CircularQueue(int capacity) : data(capacity), front_idx(0), back_idx(-1), sz(0), cap(capacity) {}

    bool push(int val) {
        if (sz == cap) return false;
        back_idx = (back_idx + 1) % cap;
        data[back_idx] = val;
        sz++;
        return true;
    }
    int front() { return data[front_idx]; }
    int back() { return data[back_idx]; }
    bool pop() {
        if (sz == 0) return false;
        front_idx = (front_idx + 1) % cap;
        sz--;
        return true;
    }
    bool empty() { return sz == 0; }
    bool full() { return sz == cap; }
    int size() { return sz; }
};

// Queue using Two Stacks — Amortized O(1)
struct QueueUsingStacks {
    stack<int> in_stack, out_stack;
    void push(int val) { in_stack.push(val); }
    void transfer() {
        if (out_stack.empty())
            while (!in_stack.empty()) { out_stack.push(in_stack.top()); in_stack.pop(); }
    }
    int front() { transfer(); return out_stack.top(); }
    void pop() { transfer(); out_stack.pop(); }
    bool empty() { return in_stack.empty() && out_stack.empty(); }
};

// ═══════════════════════════════════════════════════════════
// 8. HASH MAP — Chaining + Open Addressing
// ═══════════════════════════════════════════════════════════

// Chaining-based HashMap
struct HashMapChaining {
    static const int BUCKET_COUNT = 1 << 16; // 65536
    struct Entry { int key, val; };
    vector<Entry> buckets[BUCKET_COUNT];

    int hash(int key) { return ((unsigned)key * 2654435761u) >> 16 & (BUCKET_COUNT - 1); }

    void put(int key, int val) {
        int h = hash(key);
        for (auto& e : buckets[h]) {
            if (e.key == key) { e.val = val; return; }
        }
        buckets[h].push_back({key, val});
    }

    int get(int key, int default_val = -1) {
        int h = hash(key);
        for (auto& e : buckets[h])
            if (e.key == key) return e.val;
        return default_val;
    }

    bool contains(int key) {
        int h = hash(key);
        for (auto& e : buckets[h])
            if (e.key == key) return true;
        return false;
    }

    void erase(int key) {
        int h = hash(key);
        auto& b = buckets[h];
        for (int i = 0; i < (int)b.size(); i++) {
            if (b[i].key == key) { b.erase(b.begin() + i); return; }
        }
    }
};

// Open Addressing HashMap (Linear Probing)
struct HashMapOpenAddr {
    static const int CAP = 1 << 17; // must be power of 2
    static const int EMPTY = INT_MIN;
    int keys[CAP], vals[CAP];
    bool used[CAP];

    HashMapOpenAddr() { memset(used, false, sizeof(used)); }

    int hash(int key) { return ((unsigned)key * 2654435761u) >> 15 & (CAP - 1); }

    void put(int key, int val) {
        int h = hash(key);
        while (used[h] && keys[h] != key) h = (h + 1) & (CAP - 1);
        keys[h] = key; vals[h] = val; used[h] = true;
    }

    int get(int key, int default_val = -1) {
        int h = hash(key);
        while (used[h]) {
            if (keys[h] == key) return vals[h];
            h = (h + 1) & (CAP - 1);
        }
        return default_val;
    }

    bool contains(int key) {
        int h = hash(key);
        while (used[h]) {
            if (keys[h] == key) return true;
            h = (h + 1) & (CAP - 1);
        }
        return false;
    }
};

// ═══════════════════════════════════════════════════════════
// 9. LRU CACHE — O(1) get/put
// ═══════════════════════════════════════════════════════════
struct LRUCache {
    int capacity;
    list<pair<int,int>> dll; // {key, val}, front = most recent
    unordered_map<int, list<pair<int,int>>::iterator> mp;

    LRUCache(int cap) : capacity(cap) {}

    int get(int key) {
        if (mp.find(key) == mp.end()) return -1;
        dll.splice(dll.begin(), dll, mp[key]); // move to front
        return mp[key]->second;
    }

    void put(int key, int val) {
        if (mp.count(key)) {
            dll.splice(dll.begin(), dll, mp[key]);
            mp[key]->second = val;
            return;
        }
        if ((int)dll.size() == capacity) {
            mp.erase(dll.back().first);
            dll.pop_back();
        }
        dll.push_front({key, val});
        mp[key] = dll.begin();
    }
};

// ═══════════════════════════════════════════════════════════
// 10. INTERVAL TREE — O(log N) query for overlapping intervals
// ═══════════════════════════════════════════════════════════
struct IntervalTree {
    struct Interval { int lo, hi; };
    struct Node {
        Interval interval;
        int max_hi;
        Node *left, *right;
        Node(Interval i) : interval(i), max_hi(i.hi), left(nullptr), right(nullptr) {}
    };
    Node* root = nullptr;

    Node* insert(Node* node, Interval i) {
        if (!node) return new Node(i);
        if (i.lo < node->interval.lo)
            node->left = insert(node->left, i);
        else
            node->right = insert(node->right, i);
        node->max_hi = max(node->max_hi, i.hi);
        return node;
    }
    void insert(int lo, int hi) { root = insert(root, {lo, hi}); }

    bool overlaps(Interval a, Interval b) { return a.lo <= b.hi && b.lo <= a.hi; }

    // Find ANY interval that overlaps with [lo, hi]
    Interval* search(Node* node, Interval i) {
        if (!node) return nullptr;
        if (overlaps(node->interval, i)) return &node->interval;
        if (node->left && node->left->max_hi >= i.lo)
            return search(node->left, i);
        return search(node->right, i);
    }
    Interval* search(int lo, int hi) { return search(root, {lo, hi}); }

    // Find ALL overlapping intervals
    void search_all(Node* node, Interval i, vector<Interval>& result) {
        if (!node) return;
        if (overlaps(node->interval, i)) result.push_back(node->interval);
        if (node->left && node->left->max_hi >= i.lo)
            search_all(node->left, i, result);
        search_all(node->right, i, result);
    }
    vector<Interval> search_all(int lo, int hi) {
        vector<Interval> result;
        search_all(root, {lo, hi}, result);
        return result;
    }
};
