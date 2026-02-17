/*
 * ============================================================
 *         MATH & NUMBER THEORY — CP TEMPLATE
 * ============================================================
 *  Topics covered:
 *    1.  Fast Exponentiation (mod pow)
 *    2.  Modular Inverse
 *    3.  Modular Arithmetic Class (Mint)
 *    4.  Sieve of Eratosthenes + Smallest Prime Factor
 *    5.  Prime Factorization
 *    6.  Miller-Rabin Primality Test
 *    7.  Pollard's Rho Factorization
 *    8.  Euler's Totient Function
 *    9.  Combinatorics (nCr, nPr, Catalan, Derangements)
 *   10.  Chinese Remainder Theorem
 *   11.  Extended Euclidean Algorithm
 *   12.  Linear Diophantine Equations
 *   13.  FFT / NTT (for polynomial multiplication)
 *   14.  Lucas Theorem (nCr mod small prime)
 *   15.  Gaussian Elimination
 *   16.  Floor / Ceil division
 *   17.  Number of Divisors / Sum of Divisors
 *   18.  Mobius Function + Sieve
 *   19.  Discrete Log (Baby-Step Giant-Step)
 * ============================================================
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef __int128 lll;
const int MOD = 1e9 + 7;
const int MOD2 = 998244353;

// ═══════════════════════════════════════════════════════════
// 1. FAST EXPONENTIATION — O(log exp)
// ═══════════════════════════════════════════════════════════
ll power(ll base, ll exp, ll mod = MOD) {
    ll result = 1; base %= mod;
    if (base < 0) base += mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════
// 2. MODULAR INVERSE — O(log mod)
// ═══════════════════════════════════════════════════════════
// Using Fermat's little theorem (mod must be prime)
ll modinv(ll a, ll mod = MOD) { return power(a, mod - 2, mod); }

// Using extended GCD (works for any coprime a, mod)
ll modinv_ext(ll a, ll mod) {
    ll g, x, y;
    auto extgcd = [](ll a, ll b, ll& g, ll& x, ll& y) {
        function<void(ll,ll,ll&,ll&,ll&)> solve = [&](ll a, ll b, ll& g, ll& x, ll& y) {
            if (b == 0) { g = a; x = 1; y = 0; return; }
            solve(b, a % b, g, y, x);
            y -= (a / b) * x;
        };
        solve(a, b, g, x, y);
    };
    extgcd(a, mod, g, x, y);
    return (x % mod + mod) % mod;
}

// ═══════════════════════════════════════════════════════════
// 3. MODULAR ARITHMETIC CLASS (Mint) — auto-mod everything
// ═══════════════════════════════════════════════════════════
template<int MOD_VAL>
struct Mint {
    int val;
    Mint(ll v = 0) : val(v % MOD_VAL) { if (val < 0) val += MOD_VAL; }
    Mint operator+(const Mint& o) const { return Mint(val + o.val); }
    Mint operator-(const Mint& o) const { return Mint(val - o.val + MOD_VAL); }
    Mint operator*(const Mint& o) const { return Mint((ll)val * o.val); }
    Mint operator/(const Mint& o) const { return *this * o.inv(); }
    Mint& operator+=(const Mint& o) { return *this = *this + o; }
    Mint& operator-=(const Mint& o) { return *this = *this - o; }
    Mint& operator*=(const Mint& o) { return *this = *this * o; }
    Mint& operator/=(const Mint& o) { return *this = *this / o; }
    bool operator==(const Mint& o) const { return val == o.val; }
    bool operator!=(const Mint& o) const { return val != o.val; }
    Mint pow(ll e) const {
        Mint res(1), b(val);
        for (; e > 0; e >>= 1) { if (e & 1) res *= b; b *= b; }
        return res;
    }
    Mint inv() const { return pow(MOD_VAL - 2); }
    friend ostream& operator<<(ostream& os, const Mint& m) { return os << m.val; }
    friend istream& operator>>(istream& is, Mint& m) { ll v; is >> v; m = Mint(v); return is; }
};
using mint = Mint<MOD>;
// using mint2 = Mint<MOD2>;

// ═══════════════════════════════════════════════════════════
// 4. SIEVE OF ERATOSTHENES + SPF — O(N log log N)
// ═══════════════════════════════════════════════════════════
const int MAXN = 1e7 + 5;
bool is_prime_arr[MAXN];
int spf[MAXN]; // smallest prime factor
vector<int> primes;

void sieve(int n = MAXN - 1) {
    fill(is_prime_arr, is_prime_arr + n + 1, true);
    is_prime_arr[0] = is_prime_arr[1] = false;
    for (int i = 2; i <= n; i++) {
        if (is_prime_arr[i]) {
            primes.push_back(i);
            spf[i] = i;
            for (ll j = (ll)i * i; j <= n; j += i) {
                is_prime_arr[j] = false;
                if (spf[j] == 0) spf[j] = i;
            }
        }
    }
}

// Linear Sieve — O(N), each composite marked exactly once
void linear_sieve(int n = MAXN - 1) {
    for (int i = 2; i <= n; i++) {
        if (spf[i] == 0) { spf[i] = i; primes.push_back(i); }
        for (int j = 0; j < (int)primes.size() && primes[j] <= spf[i] && (ll)i * primes[j] <= n; j++)
            spf[i * primes[j]] = primes[j];
    }
}

// ═══════════════════════════════════════════════════════════
// 5. PRIME FACTORIZATION — O(sqrt(N)) or O(log N) with SPF
// ═══════════════════════════════════════════════════════════
// Trial division — O(sqrt(N))
map<ll, int> factorize(ll n) {
    map<ll, int> factors;
    for (ll d = 2; d * d <= n; d++) {
        while (n % d == 0) { factors[d]++; n /= d; }
    }
    if (n > 1) factors[n]++;
    return factors;
}

// Using SPF — O(log N) after sieve
map<int, int> factorize_spf(int n) {
    map<int, int> factors;
    while (n > 1) {
        factors[spf[n]]++;
        n /= spf[n];
    }
    return factors;
}

// ═══════════════════════════════════════════════════════════
// 6. MILLER-RABIN PRIMALITY TEST — O(k * log² N)
// ═══════════════════════════════════════════════════════════
// Deterministic for N < 3.317×10^24 with these witnesses
bool miller_rabin(ll n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    ll d = n - 1; int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }
    auto witness = [&](ll a) -> bool {
        ll x = 1; ll tmp = d;
        // compute a^d mod n
        ll base = a % n;
        ll res = 1;
        while (tmp > 0) {
            if (tmp & 1) res = (lll)res * base % n;
            base = (lll)base * base % n;
            tmp >>= 1;
        }
        x = res;
        if (x == 1 || x == n - 1) return false;
        for (int i = 0; i < r - 1; i++) {
            x = (lll)x * x % n;
            if (x == n - 1) return false;
        }
        return true; // composite
    };
    for (ll a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a) return true;
        if (witness(a)) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════
// 7. POLLARD'S RHO FACTORIZATION — expected O(N^(1/4))
// ═══════════════════════════════════════════════════════════
ll pollard_rho(ll n) {
    if (n % 2 == 0) return 2;
    ll x = rand() % (n - 2) + 2;
    ll y = x, c = rand() % (n - 1) + 1, d = 1;
    while (d == 1) {
        x = ((lll)x * x + c) % n;
        y = ((lll)y * y + c) % n;
        y = ((lll)y * y + c) % n;
        d = __gcd(abs(x - y), n);
    }
    return d == n ? pollard_rho(n) : d;
}

map<ll, int> factorize_large(ll n) {
    if (n == 1) return {};
    if (miller_rabin(n)) return {{n, 1}};
    ll d = pollard_rho(n);
    auto f1 = factorize_large(d);
    auto f2 = factorize_large(n / d);
    for (auto& [p, c] : f2) f1[p] += c;
    return f1;
}

// ═══════════════════════════════════════════════════════════
// 8. EULER'S TOTIENT FUNCTION — O(sqrt(N)) or O(N) sieve
// ═══════════════════════════════════════════════════════════
ll euler_phi(ll n) {
    ll result = n;
    for (ll d = 2; d * d <= n; d++) {
        if (n % d == 0) {
            while (n % d == 0) n /= d;
            result -= result / d;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}

// Sieve variant — compute phi for all 1..n
vector<int> phi_sieve(int n) {
    vector<int> phi(n + 1);
    iota(phi.begin(), phi.end(), 0);
    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) { // i is prime
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
        }
    }
    return phi;
}

// ═══════════════════════════════════════════════════════════
// 9. COMBINATORICS — nCr, nPr, Catalan, Derangements
// ═══════════════════════════════════════════════════════════
struct Combo {
    int n;
    vector<ll> fact, inv_fact;

    Combo(int n = 2e6) : n(n), fact(n + 1), inv_fact(n + 1) {
        fact[0] = 1;
        for (int i = 1; i <= n; i++) fact[i] = fact[i-1] * i % MOD;
        inv_fact[n] = power(fact[n], MOD - 2, MOD);
        for (int i = n - 1; i >= 0; i--) inv_fact[i] = inv_fact[i+1] * (i+1) % MOD;
    }

    ll nCr(int n, int r) {
        if (r < 0 || r > n) return 0;
        return fact[n] % MOD * inv_fact[r] % MOD * inv_fact[n-r] % MOD;
    }

    ll nPr(int n, int r) {
        if (r < 0 || r > n) return 0;
        return fact[n] % MOD * inv_fact[n-r] % MOD;
    }

    // Catalan(n) = C(2n, n) / (n + 1)
    ll catalan(int n) {
        return nCr(2*n, n) % MOD * power(n + 1, MOD - 2, MOD) % MOD;
    }

    // Derangements D(n) = (n-1) * (D(n-1) + D(n-2))
    ll derangement(int n) {
        if (n == 0) return 1;
        if (n == 1) return 0;
        ll a = 1, b = 0; // D(0), D(1)
        for (int i = 2; i <= n; i++) {
            ll c = (ll)(i - 1) % MOD * ((a + b) % MOD) % MOD;
            a = b; b = c;
        }
        return b;
    }

    // Stars and Bars: ways to put n identical balls in k distinct boxes
    ll stars_and_bars(int n, int k) { return nCr(n + k - 1, k - 1); }
};

// ═══════════════════════════════════════════════════════════
// 10. CHINESE REMAINDER THEOREM — O(N log M)
// ═══════════════════════════════════════════════════════════
// Solve system: x ≡ a_i (mod m_i) for pairwise coprime m_i
// Returns {x, M} where M = product of all m_i
pair<ll, ll> crt(const vector<ll>& a, const vector<ll>& m) {
    ll M = 1, x = 0;
    for (ll mi : m) M *= mi;
    for (int i = 0; i < (int)a.size(); i++) {
        ll Mi = M / m[i];
        ll yi = power(Mi, m[i] - 2, m[i]); // modinv(Mi, m[i])
        x = (x + (lll)a[i] % M * Mi % M * yi % M) % M;
    }
    return {(x % M + M) % M, M};
}

// General CRT (moduli need NOT be coprime)
// Returns {x, lcm} or {-1, -1} if no solution
pair<ll, ll> crt_general(ll a1, ll m1, ll a2, ll m2) {
    ll g = __gcd(m1, m2);
    if ((a2 - a1) % g != 0) return {-1, -1};
    ll lcm = m1 / g * m2;
    ll diff = (a2 - a1) / g;
    ll inv = power(m1 / g, m2 / g - 2, m2 / g); // modinv
    ll x = (a1 + (lll)m1 * diff % lcm * inv % lcm) % lcm;
    return {(x + lcm) % lcm, lcm};
}

// ═══════════════════════════════════════════════════════════
// 11. EXTENDED EUCLIDEAN — ax + by = gcd(a, b)
// ═══════════════════════════════════════════════════════════
ll ext_gcd(ll a, ll b, ll& x, ll& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    ll x1, y1;
    ll g = ext_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// ═══════════════════════════════════════════════════════════
// 12. LINEAR DIOPHANTINE — ax + by = c
// ═══════════════════════════════════════════════════════════
// Returns {x0, y0} such that a*x0 + b*y0 = c, or {-1,-1} if impossible
pair<ll,ll> diophantine(ll a, ll b, ll c) {
    ll x, y;
    ll g = ext_gcd(abs(a), abs(b), x, y);
    if (c % g != 0) return {-1, -1};
    x *= c / g; y *= c / g;
    if (a < 0) x = -x;
    if (b < 0) y = -y;
    return {x, y};
    // General solution: x = x0 + k*(b/g), y = y0 - k*(a/g) for any integer k
}

// ═══════════════════════════════════════════════════════════
// 13. NTT (Number Theoretic Transform) — O(N log N)
// ═══════════════════════════════════════════════════════════
// Works with MOD = 998244353 (primitive root g = 3)
void ntt(vector<ll>& a, bool inv = false) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        ll w = inv ? power(3, MOD2 - 1 - (MOD2 - 1) / len, MOD2) : power(3, (MOD2 - 1) / len, MOD2);
        for (int i = 0; i < n; i += len) {
            ll wn = 1;
            for (int j = 0; j < len / 2; j++) {
                ll u = a[i + j], v = a[i + j + len/2] * wn % MOD2;
                a[i + j] = (u + v) % MOD2;
                a[i + j + len/2] = (u - v + MOD2) % MOD2;
                wn = wn * w % MOD2;
            }
        }
    }
    if (inv) {
        ll n_inv = power(n, MOD2 - 2, MOD2);
        for (ll& x : a) x = x * n_inv % MOD2;
    }
}

// Polynomial multiplication — result[k] = sum of a[i]*b[j] where i+j=k
vector<ll> poly_multiply(vector<ll> a, vector<ll> b) {
    int result_sz = a.size() + b.size() - 1;
    int n = 1;
    while (n < result_sz) n <<= 1;
    a.resize(n); b.resize(n);
    ntt(a); ntt(b);
    for (int i = 0; i < n; i++) a[i] = a[i] * b[i] % MOD2;
    ntt(a, true);
    a.resize(result_sz);
    return a;
}

// ═══════════════════════════════════════════════════════════
// 14. LUCAS THEOREM — nCr mod p (small prime p)
// ═══════════════════════════════════════════════════════════
ll lucas(ll n, ll r, ll p) {
    if (r == 0) return 1;
    return lucas(n / p, r / p, p) * [](ll n, ll r, ll p) -> ll {
        if (r > n) return 0;
        ll num = 1, den = 1;
        for (ll i = 0; i < r; i++) {
            num = num * ((n - i) % p) % p;
            den = den * ((i + 1) % p) % p;
        }
        return num % p * power(den, p - 2, p) % p;
    }(n % p, r % p, p) % p;
}

// ═══════════════════════════════════════════════════════════
// 15. GAUSSIAN ELIMINATION — O(N² * M)
// ═══════════════════════════════════════════════════════════
// Solves Ax = b. Returns rank. Solution in b.
// Returns -1 if no solution.
int gauss(vector<vector<double>>& a, vector<double>& b) {
    int n = a.size(), m = a[0].size();
    vector<int> where(m, -1);
    int rank = 0;
    for (int col = 0; col < m && rank < n; col++) {
        int pivot = rank;
        for (int i = rank + 1; i < n; i++)
            if (abs(a[i][col]) > abs(a[pivot][col])) pivot = i;
        if (abs(a[pivot][col]) < 1e-9) continue;
        swap(a[rank], a[pivot]); swap(b[rank], b[pivot]);
        where[col] = rank;
        for (int i = 0; i < n; i++) {
            if (i == rank) continue;
            double c = a[i][col] / a[rank][col];
            for (int j = col; j < m; j++) a[i][j] -= c * a[rank][j];
            b[i] -= c * b[rank];
        }
        rank++;
    }
    // Check for inconsistency
    for (int i = rank; i < n; i++)
        if (abs(b[i]) > 1e-9) return -1;
    // Extract solution
    for (int j = 0; j < m; j++)
        if (where[j] != -1) b[where[j]] /= a[where[j]][j];
    return rank;
}

// Gaussian Elimination over GF(2) — for XOR problems
int gauss_xor(vector<ll>& basis) {
    int rank = 0;
    sort(basis.rbegin(), basis.rend());
    vector<ll> reduced;
    for (ll x : basis) {
        for (ll b : reduced) x = min(x, x ^ b);
        if (x > 0) reduced.push_back(x);
    }
    basis = reduced;
    return basis.size();
}

// ═══════════════════════════════════════════════════════════
// 16. FLOOR / CEIL DIVISION — handles negatives correctly
// ═══════════════════════════════════════════════════════════
ll floor_div(ll a, ll b) { return a / b - (a % b != 0 && (a ^ b) < 0); }
ll ceil_div(ll a, ll b)  { return a / b + (a % b != 0 && (a ^ b) > 0); }

// ═══════════════════════════════════════════════════════════
// 17. DIVISOR FUNCTIONS — O(sqrt(N))
// ═══════════════════════════════════════════════════════════
int count_divisors(ll n) {
    int cnt = 0;
    for (ll d = 1; d * d <= n; d++)
        if (n % d == 0) cnt += (d * d == n) ? 1 : 2;
    return cnt;
}

ll sum_divisors(ll n) {
    ll sum = 0;
    for (ll d = 1; d * d <= n; d++)
        if (n % d == 0) { sum += d; if (d != n / d) sum += n / d; }
    return sum;
}

// All divisors of n, sorted
vector<ll> get_divisors(ll n) {
    vector<ll> divs;
    for (ll d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            divs.push_back(d);
            if (d != n / d) divs.push_back(n / d);
        }
    }
    sort(divs.begin(), divs.end());
    return divs;
}

// ═══════════════════════════════════════════════════════════
// 18. MOBIUS FUNCTION + SIEVE
// ═══════════════════════════════════════════════════════════
vector<int> mobius_sieve(int n) {
    vector<int> mu(n + 1, 1);
    vector<bool> not_prime(n + 1, false);
    vector<int> pr;
    for (int i = 2; i <= n; i++) {
        if (!not_prime[i]) { pr.push_back(i); mu[i] = -1; }
        for (int j = 0; j < (int)pr.size() && (ll)i * pr[j] <= n; j++) {
            not_prime[i * pr[j]] = true;
            if (i % pr[j] == 0) { mu[i * pr[j]] = 0; break; }
            mu[i * pr[j]] = -mu[i];
        }
    }
    return mu;
}

// ═══════════════════════════════════════════════════════════
// 19. DISCRETE LOG — Baby-Step Giant-Step — O(sqrt(M))
// ═══════════════════════════════════════════════════════════
// Find smallest x >= 0 such that a^x ≡ b (mod m), or -1 if no solution
ll discrete_log(ll a, ll b, ll m) {
    a %= m; b %= m;
    if (b == 1 % m) return 0;
    ll sq = (ll)ceil(sqrt((double)m));
    unordered_map<ll, ll> table;
    // Baby step: store a^j for j = 0..sq-1
    ll cur = b;
    for (ll j = 0; j < sq; j++) {
        table[cur] = j;
        cur = (lll)cur * a % m;
    }
    // Giant step: a^(sq)
    ll giant = power(a, sq, m);
    cur = 1;
    for (ll i = 0; i <= sq; i++) {
        if (table.count(cur)) {
            ll ans = i * sq - table[cur];
            if (ans >= 0) return ans;
        }
        cur = (lll)cur * giant % m;
    }
    return -1;
}

/*
 * ══════════════════════════════════════
 *  MATH CHEAT SHEET
 * ══════════════════════════════════════
 *
 * Key identities:
 *   a^(p-1) ≡ 1 (mod p)                 — Fermat's little theorem
 *   a^phi(m) ≡ 1 (mod m) if gcd(a,m)=1  — Euler's theorem
 *   nCr mod p → Lucas theorem (p small)
 *   sum_{d|n} phi(d) = n
 *   sum_{d|n} mu(d) = [n == 1]
 *
 * Common MODs:
 *   1e9 + 7  (prime, good for most problems)
 *   998244353 (= 119 * 2^23 + 1, good for NTT, primitive root = 3)
 *
 * Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, ...
 *   C(n) = C(2n,n)/(n+1) = sum_{i=0..n-1} C(i)*C(n-1-i)
 *   Applications: valid parentheses, BST count, triangulations
 *
 * Inclusion-Exclusion:
 *   |A1 ∪ A2 ∪ ... ∪ An| = Σ|Ai| - Σ|Ai∩Aj| + Σ|Ai∩Aj∩Ak| - ...
 *   Often combined with Mobius function
 */
