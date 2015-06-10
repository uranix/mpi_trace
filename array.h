#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <istream>
#include <cstddef>
#include <vector>

namespace fort {

class range {
    ptrdiff_t beg;
    size_t dim;
public:
    range() : beg(0), dim(0) { }
    range(ptrdiff_t n) : beg(1), dim(n) { }
    range(ptrdiff_t beg, ptrdiff_t end) : beg(beg), dim(end - beg + 1) { }
    ptrdiff_t zero_based(ptrdiff_t idx) const { return idx - beg; }
    size_t size() const { return dim; }
};

template<int n, typename T>
class array {
    std::vector<T> buf;
    range dims[n];

    template<int m>
    ptrdiff_t idx(ptrdiff_t i) const {
        static_assert(m == n - 1, "Index parameters mismatch array dimension");
        return dims[m].zero_based(i);
    }

    template<int m, typename ...Args>
    ptrdiff_t idx(ptrdiff_t i, const Args &...args) const {
        return dims[m].zero_based(i) + dims[m].size() * idx<m + 1>(args...);
    }
    template<int m>
    void choke(const range &dim) {
        static_assert(m == n - 1, "Constructor parameters mismatch array dimension");
        dims[m] = dim;
    }
    template<int m, typename ...Args>
    void choke(const range &dim, const Args &...args) {
        dims[m] = dim;
        choke<m + 1>(args...);
    }
public:
    template<typename ...Args>
    array(const Args &...args) {
        choke<0>(args...);
        size_t sz = dims[0].size();
        for (int i = 1; i < n; i++)
            sz *= dims[i].size();
        buf.resize(sz);
    }
    std::vector<T> &container() { return buf; }
    const std::vector<T> &container() const { return buf; }
    T *data() { return buf.data(); }
    const T *data() const { return buf.data(); }
    template<typename ...Args>
    const T &operator()(const Args &...args) const {
        return buf[idx<0>(args...)];
    }
    template<typename ...Args>
    T &operator()(const Args &...args) {
        return buf[idx<0>(args...)];
    }
};

template<int n, typename T>
inline std::istream &operator>>(std::istream &i, array<n, T> &z) {
    auto &c = z.container();
    i.read(reinterpret_cast<char *>(c.data()), c.size() * sizeof(float));
    return i;
}
}
#endif

