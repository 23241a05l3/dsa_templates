/*
 * ============================================================
 *     SORTING, SEARCHING & TWO POINTERS — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Bubble Sort — O(N²)
 *    2.  Selection Sort — O(N²)
 *    3.  Insertion Sort — O(N²)
 *    4.  Merge Sort — O(N log N) + Inversion Count
 *    5.  Quick Sort + Randomized — O(N log N) avg
 *    6.  Heap Sort — O(N log N)
 *    7.  Counting Sort — O(N + K)
 *    8.  Radix Sort — O(N * D)
 *    9.  Binary Search — all patterns
 *   10.  Two Pointers — all patterns
 *   11.  Sliding Window — fixed & variable
 *   12.  Three-way Partition (Dutch National Flag)
 *   13.  Quick Select (Kth element) — O(N) avg
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ═══════════════════════════════════════════════════════════
// 1. BUBBLE SORT — O(N²)
// ═══════════════════════════════════════════════════════════
void bubble_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j+1]) { swap(a[j], a[j+1]); swapped = true; }
        }
        if (!swapped) break; // already sorted
    }
}

// ═══════════════════════════════════════════════════════════
// 2. SELECTION SORT — O(N²)
// ═══════════════════════════════════════════════════════════
void selection_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int mn = i;
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[mn]) mn = j;
        swap(a[i], a[mn]);
    }
}

// ═══════════════════════════════════════════════════════════
// 3. INSERTION SORT — O(N²) — good for nearly sorted
// ═══════════════════════════════════════════════════════════
void insertion_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i], j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
        a[j+1] = key;
    }
}

// ═══════════════════════════════════════════════════════════
// 4. MERGE SORT — O(N log N) + INVERSION COUNT
// ═══════════════════════════════════════════════════════════
ll merge_count(vector<int>& a, int l, int r) {
    if (l >= r) return 0;
    int mid = (l + r) / 2;
    ll inv = merge_count(a, l, mid) + merge_count(a, mid + 1, r);
    vector<int> tmp;
    int i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) tmp.push_back(a[i++]);
        else { tmp.push_back(a[j++]); inv += mid - i + 1; }
    }
    while (i <= mid) tmp.push_back(a[i++]);
    while (j <= r) tmp.push_back(a[j++]);
    for (int k = l; k <= r; k++) a[k] = tmp[k - l];
    return inv;
}

void merge_sort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int mid = (l + r) / 2;
    merge_sort(a, l, mid);
    merge_sort(a, mid + 1, r);
    vector<int> tmp;
    int i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) tmp.push_back(a[i++]);
        else tmp.push_back(a[j++]);
    }
    while (i <= mid) tmp.push_back(a[i++]);
    while (j <= r) tmp.push_back(a[j++]);
    for (int k = l; k <= r; k++) a[k] = tmp[k - l];
}
void merge_sort(vector<int>& a) { merge_sort(a, 0, a.size() - 1); }

// Count inversions
ll count_inversions(vector<int> a) { return merge_count(a, 0, a.size() - 1); }

// ═══════════════════════════════════════════════════════════
// 5. QUICK SORT — O(N log N) average, O(N²) worst
// ═══════════════════════════════════════════════════════════
mt19937 rng_qs(chrono::steady_clock::now().time_since_epoch().count());

int partition(vector<int>& a, int lo, int hi) {
    // Random pivot for anti-hack
    int pivot_idx = uniform_int_distribution<int>(lo, hi)(rng_qs);
    swap(a[pivot_idx], a[hi]);
    int pivot = a[hi], i = lo;
    for (int j = lo; j < hi; j++) {
        if (a[j] < pivot) swap(a[i++], a[j]);
    }
    swap(a[i], a[hi]);
    return i;
}

void quick_sort(vector<int>& a, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(a, lo, hi);
    quick_sort(a, lo, p - 1);
    quick_sort(a, p + 1, hi);
}
void quick_sort(vector<int>& a) { quick_sort(a, 0, a.size() - 1); }

// ═══════════════════════════════════════════════════════════
// 6. HEAP SORT — O(N log N) in-place
// ═══════════════════════════════════════════════════════════
void heap_sort(vector<int>& a) {
    int n = a.size();
    // Build max heap
    auto sift_down = [&](int i, int sz) {
        while (true) {
            int largest = i, l = 2*i+1, r = 2*i+2;
            if (l < sz && a[l] > a[largest]) largest = l;
            if (r < sz && a[r] > a[largest]) largest = r;
            if (largest == i) break;
            swap(a[i], a[largest]);
            i = largest;
        }
    };
    for (int i = n/2 - 1; i >= 0; i--) sift_down(i, n);
    // Extract elements
    for (int i = n - 1; i > 0; i--) {
        swap(a[0], a[i]);
        sift_down(0, i);
    }
}

// ═══════════════════════════════════════════════════════════
// 7. COUNTING SORT — O(N + K) — for integers in range [0, K)
// ═══════════════════════════════════════════════════════════
void counting_sort(vector<int>& a) {
    if (a.empty()) return;
    int mn = *min_element(a.begin(), a.end());
    int mx = *max_element(a.begin(), a.end());
    int range = mx - mn + 1;
    vector<int> count(range, 0), output(a.size());
    for (int x : a) count[x - mn]++;
    for (int i = 1; i < range; i++) count[i] += count[i-1];
    for (int i = a.size() - 1; i >= 0; i--)
        output[--count[a[i] - mn]] = a[i];
    a = output;
}

// ═══════════════════════════════════════════════════════════
// 8. RADIX SORT — O(N * D) — for non-negative integers
// ═══════════════════════════════════════════════════════════
void radix_sort(vector<int>& a) {
    if (a.empty()) return;
    int mx = *max_element(a.begin(), a.end());
    for (int exp = 1; mx / exp > 0; exp *= 10) {
        int n = a.size();
        vector<int> output(n), count(10, 0);
        for (int x : a) count[(x / exp) % 10]++;
        for (int i = 1; i < 10; i++) count[i] += count[i-1];
        for (int i = n - 1; i >= 0; i--)
            output[--count[(a[i] / exp) % 10]] = a[i];
        a = output;
    }
}

// ═══════════════════════════════════════════════════════════
// 9. BINARY SEARCH — All Patterns
// ═══════════════════════════════════════════════════════════

// Pattern 1: Find exact value (returns index or -1)
int binary_search_exact(const vector<int>& a, int target) {
    int lo = 0, hi = (int)a.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == target) return mid;
        if (a[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// Pattern 2: First position >= target (lower_bound)
int lower_bound_manual(const vector<int>& a, int target) {
    int lo = 0, hi = a.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Pattern 3: First position > target (upper_bound)
int upper_bound_manual(const vector<int>& a, int target) {
    int lo = 0, hi = a.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Pattern 4: Last position <= target
int last_leq(const vector<int>& a, int target) {
    return upper_bound_manual(a, target) - 1;
}

// Pattern 5: Binary search on answer (find min x such that check(x) is true)
// check is monotone: false,...,false,true,...,true
ll bs_answer_min(ll lo, ll hi, function<bool(ll)> check) {
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (check(mid)) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Pattern 6: Binary search on answer (find max x such that check(x) is true)
// check is monotone: true,...,true,false,...,false
ll bs_answer_max(ll lo, ll hi, function<bool(ll)> check) {
    while (lo < hi) {
        ll mid = lo + (hi - lo + 1) / 2; // round UP
        if (check(mid)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Pattern 7: Binary search on real-valued answer
double bs_real(double lo, double hi, function<bool(double)> check, int iter = 200) {
    for (int i = 0; i < iter; i++) {
        double mid = (lo + hi) / 2;
        if (check(mid)) hi = mid;
        else lo = mid;
    }
    return (lo + hi) / 2;
}

// Pattern 8: Binary search on rotated sorted array
int search_rotated(const vector<int>& a, int target) {
    int lo = 0, hi = (int)a.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == target) return mid;
        if (a[lo] <= a[mid]) { // left half sorted
            if (a[lo] <= target && target < a[mid]) hi = mid - 1;
            else lo = mid + 1;
        } else { // right half sorted
            if (a[mid] < target && target <= a[hi]) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    return -1;
}

// Pattern 9: Peak element (bitonic search)
int find_peak(const vector<int>& a) {
    int lo = 0, hi = (int)a.size() - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < a[mid + 1]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Pattern 10: Median of two sorted arrays — O(log(min(N,M)))
double median_two_sorted(const vector<int>& a, const vector<int>& b) {
    if (a.size() > b.size()) return median_two_sorted(b, a);
    int n = a.size(), m = b.size(), half = (n + m + 1) / 2;
    int lo = 0, hi = n;
    while (lo <= hi) {
        int i = (lo + hi) / 2, j = half - i;
        int left_a = (i > 0) ? a[i-1] : INT_MIN;
        int right_a = (i < n) ? a[i] : INT_MAX;
        int left_b = (j > 0) ? b[j-1] : INT_MIN;
        int right_b = (j < m) ? b[j] : INT_MAX;
        if (left_a <= right_b && left_b <= right_a) {
            if ((n + m) % 2 == 1) return max(left_a, left_b);
            return (max(left_a, left_b) + min(right_a, right_b)) / 2.0;
        }
        if (left_a > right_b) hi = i - 1;
        else lo = i + 1;
    }
    return -1; // should never reach
}

// ═══════════════════════════════════════════════════════════
// 10. TWO POINTERS — All Patterns
// ═══════════════════════════════════════════════════════════

// Pattern 1: Two Sum (sorted array)
pair<int,int> two_sum_sorted(const vector<int>& a, int target) {
    int l = 0, r = (int)a.size() - 1;
    while (l < r) {
        int sum = a[l] + a[r];
        if (sum == target) return {l, r};
        if (sum < target) l++;
        else r--;
    }
    return {-1, -1};
}

// Pattern 2: Remove duplicates in-place (sorted)
int remove_duplicates(vector<int>& a) {
    if (a.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < (int)a.size(); fast++)
        if (a[fast] != a[slow]) a[++slow] = a[fast];
    return slow + 1;
}

// Pattern 3: Container with most water
int max_water(const vector<int>& height) {
    int l = 0, r = (int)height.size() - 1, ans = 0;
    while (l < r) {
        ans = max(ans, min(height[l], height[r]) * (r - l));
        if (height[l] < height[r]) l++;
        else r--;
    }
    return ans;
}

// Pattern 4: Three Sum = 0
vector<vector<int>> three_sum(vector<int> a) {
    sort(a.begin(), a.end());
    vector<vector<int>> res;
    int n = a.size();
    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && a[i] == a[i-1]) continue;
        int l = i + 1, r = n - 1;
        while (l < r) {
            int sum = a[i] + a[l] + a[r];
            if (sum == 0) {
                res.push_back({a[i], a[l], a[r]});
                while (l < r && a[l] == a[l+1]) l++;
                while (l < r && a[r] == a[r-1]) r--;
                l++; r--;
            } else if (sum < 0) l++;
            else r--;
        }
    }
    return res;
}

// Pattern 5: Trapping Rain Water
int trap_water(const vector<int>& height) {
    int l = 0, r = (int)height.size() - 1;
    int left_max = 0, right_max = 0, water = 0;
    while (l < r) {
        if (height[l] < height[r]) {
            left_max = max(left_max, height[l]);
            water += left_max - height[l];
            l++;
        } else {
            right_max = max(right_max, height[r]);
            water += right_max - height[r];
            r--;
        }
    }
    return water;
}

// Pattern 6: Merge two sorted arrays in-place
void merge_sorted_inplace(vector<int>& a, int m, vector<int>& b, int n_b) {
    // a has size m + n_b, last n_b elements are empty
    int i = m - 1, j = n_b - 1, k = m + n_b - 1;
    while (i >= 0 && j >= 0) {
        if (a[i] > b[j]) a[k--] = a[i--];
        else a[k--] = b[j--];
    }
    while (j >= 0) a[k--] = b[j--];
}

// ═══════════════════════════════════════════════════════════
// 11. SLIDING WINDOW — Fixed & Variable Size
// ═══════════════════════════════════════════════════════════

// Fixed window: max sum subarray of size k
ll max_sum_fixed(const vector<int>& a, int k) {
    ll sum = 0, mx = LLONG_MIN;
    for (int i = 0; i < (int)a.size(); i++) {
        sum += a[i];
        if (i >= k) sum -= a[i - k];
        if (i >= k - 1) mx = max(mx, sum);
    }
    return mx;
}

// Variable window: smallest subarray with sum >= target
int min_subarray_sum_geq(const vector<int>& a, int target) {
    int n = a.size(), l = 0, ans = n + 1;
    ll sum = 0;
    for (int r = 0; r < n; r++) {
        sum += a[r];
        while (sum >= target) {
            ans = min(ans, r - l + 1);
            sum -= a[l++];
        }
    }
    return ans > n ? -1 : ans;
}

// Variable window: longest substring without repeating characters
int longest_unique_substr(const string& s) {
    vector<int> last(128, -1);
    int l = 0, ans = 0;
    for (int r = 0; r < (int)s.size(); r++) {
        if (last[s[r]] >= l) l = last[s[r]] + 1;
        last[s[r]] = r;
        ans = max(ans, r - l + 1);
    }
    return ans;
}

// Variable window: at most K distinct characters
int at_most_k_distinct(const string& s, int k) {
    unordered_map<char, int> freq;
    int l = 0, ans = 0;
    for (int r = 0; r < (int)s.size(); r++) {
        freq[s[r]]++;
        while ((int)freq.size() > k) {
            if (--freq[s[l]] == 0) freq.erase(s[l]);
            l++;
        }
        ans = max(ans, r - l + 1);
    }
    return ans;
}

// Count subarrays with exactly K distinct = atMost(K) - atMost(K-1)
int count_subarrays_k_distinct(const vector<int>& a, int k) {
    auto atMost = [&](int k) -> int {
        unordered_map<int, int> freq;
        int l = 0, cnt = 0;
        for (int r = 0; r < (int)a.size(); r++) {
            freq[a[r]]++;
            while ((int)freq.size() > k) {
                if (--freq[a[l]] == 0) freq.erase(a[l]);
                l++;
            }
            cnt += r - l + 1;
        }
        return cnt;
    };
    return atMost(k) - atMost(k - 1);
}

// ═══════════════════════════════════════════════════════════
// 12. THREE-WAY PARTITION (Dutch National Flag) — O(N)
// ═══════════════════════════════════════════════════════════
// Partition array into: < pivot, == pivot, > pivot
void three_way_partition(vector<int>& a, int pivot) {
    int lo = 0, mid = 0, hi = (int)a.size() - 1;
    while (mid <= hi) {
        if (a[mid] < pivot) swap(a[lo++], a[mid++]);
        else if (a[mid] > pivot) swap(a[mid], a[hi--]);
        else mid++;
    }
}

// ═══════════════════════════════════════════════════════════
// 13. QUICK SELECT — Kth smallest in O(N) average
// ═══════════════════════════════════════════════════════════
int quick_select(vector<int>& a, int lo, int hi, int k) {
    if (lo == hi) return a[lo];
    int pivot_idx = uniform_int_distribution<int>(lo, hi)(rng_qs);
    swap(a[pivot_idx], a[hi]);
    int pivot = a[hi], i = lo;
    for (int j = lo; j < hi; j++)
        if (a[j] < pivot) swap(a[i++], a[j]);
    swap(a[i], a[hi]);
    if (k == i) return a[i];
    if (k < i) return quick_select(a, lo, i - 1, k);
    return quick_select(a, i + 1, hi, k);
}
// kth_smallest (0-indexed): quick_select(a, 0, n-1, k)

/*
 * ══════════════════════════════════════
 *  SORTING ALGORITHM SELECTION GUIDE
 * ══════════════════════════════════════
 *
 * Use std::sort() in contests (introsort, O(N log N) guaranteed)
 * Use std::stable_sort() when stability needed
 * Use counting/radix sort when values have small range
 *
 * In CP, manual sorts are rarely needed. Know them for:
 *   - Inversion count → merge sort
 *   - Kth element → quick select
 *   - Custom comparators → lambda with sort()
 *
 * BINARY SEARCH PATTERNS:
 *   - Exact match → Pattern 1
 *   - lower_bound, upper_bound → Patterns 2-4
 *   - Binary search on answer → Patterns 5-6 (VERY COMMON in CP)
 *   - Rotated array → Pattern 8
 *   - Peak / bitonic → Pattern 9
 *
 * TWO POINTER PATTERNS:
 *   - Opposite ends (sum problems) → converging pointers
 *   - Same direction (window problems) → sliding window
 *   - Exactly K = atMost(K) - atMost(K-1)
 */
