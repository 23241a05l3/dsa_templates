/*
 * ============================================================
 *        COMPETITIVE PROGRAMMING MASTER TEMPLATE (C++)
 * ============================================================
 *  Author : bunny
 *  Usage  : Copy this at the top of every contest solution.
 *           Then paste specific data structures / algorithms
 *           from the individual template files as needed.
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ───────────────── Compiler Optimizations ─────────────────
#pragma GCC optimize("O2,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

// ───────────────── Type Aliases ─────────────────
typedef long long              ll;
typedef unsigned long long     ull;
typedef long double            ld;
typedef pair<int,int>          pii;
typedef pair<ll,ll>            pll;
typedef vector<int>            vi;
typedef vector<ll>             vll;
typedef vector<pii>            vpii;
typedef vector<pll>            vpll;
typedef vector<vi>             vvi;
typedef vector<vll>            vvll;
typedef vector<string>         vs;

// ───────────────── Macros ─────────────────
#define endl               '\n'
#define all(x)             (x).begin(), (x).end()
#define rall(x)            (x).rbegin(), (x).rend()
#define sz(x)              (int)(x).size()
#define pb                 push_back
#define eb                 emplace_back
#define mp                 make_pair
#define fi                 first
#define se                 second
#define lb                 lower_bound
#define ub                 upper_bound

// ───────────────── Loop Macros ─────────────────
#define FOR(i,a,b)         for(int i=(a); i<(b); ++i)
#define ROF(i,a,b)         for(int i=(b)-1; i>=(a); --i)
#define REP(i,n)           FOR(i,0,n)
#define EACH(x,a)          for(auto& x : a)

// ───────────────── Debug Macros ─────────────────
#ifdef LOCAL
    #define dbg(...)  cerr << "[" << #__VA_ARGS__ << "]: "; _dbg(__VA_ARGS__)
    template<typename T> void _dbg(T t){ cerr << t << endl; }
    template<typename T, typename... A> void _dbg(T t, A... a){ cerr << t << ", "; _dbg(a...); }
    template<typename T> void dbg_vec(const vector<T>& v){
        cerr << "["; for(int i=0;i<sz(v);i++) cerr<<(i?", ":"")<<v[i]; cerr << "]\n";
    }
#else
    #define dbg(...)          42
    #define dbg_vec(...)      42
#endif

// ───────────────── Constants ─────────────────
const int MOD  = 1e9 + 7;
const int MOD2 = 998244353;
const ll  INF  = 1e18;
const int IINF = 1e9;
const ld  EPS  = 1e-9;
const ld  PI   = acos((ld)-1);
const int dx[] = {0, 0, 1, -1, 1, 1, -1, -1};   // 4-dir + 8-dir
const int dy[] = {1, -1, 0, 0, 1, -1, 1, -1};

// ───────────────── Utility Functions ─────────────────
template<class T> bool ckmin(T& a, const T& b){ return b < a ? a = b, true : false; }
template<class T> bool ckmax(T& a, const T& b){ return a < b ? a = b, true : false; }

ll power(ll base, ll exp, ll mod = MOD) {
    ll result = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}

ll modinv(ll a, ll mod = MOD) { return power(a, mod - 2, mod); }

ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }

// ───────────────── I/O Helpers ─────────────────
template<typename T> void read(vector<T>& v) { for(auto& x : v) cin >> x; }
template<typename T> void print(const vector<T>& v, string sep = " ") {
    for(int i = 0; i < sz(v); i++) cout << v[i] << (i+1<sz(v) ? sep : "\n");
}

void YES(bool f = true) { cout << (f ? "YES" : "NO") << endl; }
void Yes(bool f = true) { cout << (f ? "Yes" : "No") << endl; }

// ───────────────── Solve Function ─────────────────
void solve() {
    /*
     * YOUR SOLUTION HERE
     */
}

// ───────────────── Main ─────────────────
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int tc = 1;
    cin >> tc;           // Comment out for single test case problems
    while (tc--) solve();

    return 0;
}
