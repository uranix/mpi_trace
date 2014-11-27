#ifndef __LEBEDEVQUAD_H__
#define __LEBEDEVQUAD_H__

#include <vector>
#include <algorithm>
#include <stdexcept>

typedef void (*Quadrature)(double *, double *, double *, double *);
struct BankEntry {
    const int precision;
    const int order;
    Quadrature quad;
    BankEntry(const int precision, const int order, Quadrature quad)
        : precision(precision), order(order), quad(quad)
    { }
};

struct LebedevQuad {
	int order;
    int prec;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> w;

    template<class Func>
    auto operator()(const Func &f) const -> decltype(f(1.0, 0.0, 0.0)) {
        typedef decltype(f(1.0, 0.0, 0.0)) V;
        V sum(w[0] * f(x[0], y[0], z[0]));
        for (int i = 1; i < order; i++)
            sum += w[i] * f(x[i], y[i], z[i]);
        return sum;
    }

	explicit LebedevQuad(const BankEntry &ent) {
        order = ent.order;
        prec = ent.precision;
        x.resize(order);
        y.resize(order);
        z.resize(order);
        w.resize(order);

        ent.quad(x.data(), y.data(), z.data(), w.data());
    }
};

struct LebedevQuadBank {
    static const LebedevQuad lookupByOrder(int order) {
        return instance()._lookupByOrder(order);
    }
    static const LebedevQuad lookupByPrecision(int prec) {
        return instance()._lookupByPrecision(prec);
    }
private:
    const LebedevQuad _lookupByOrder(int order) {
        BankEntry needle(0, order, NULL);
    	auto it = std::lower_bound(entries.begin(), entries.end(), needle,
    		[](const BankEntry &a, const BankEntry &b) -> bool { return a.order < b.order; });
		if (it == entries.end())
            throw std::invalid_argument("Quadrature order is too big");
		return LebedevQuad(*it);
    }
    const LebedevQuad _lookupByPrecision(int prec) {
        BankEntry needle(prec, 0, NULL);
		auto it = std::lower_bound(entries.begin(), entries.end(), needle,
    		[](const BankEntry &a, const BankEntry &b) -> bool { return a.precision < b.precision; });
		if (it == entries.end())
            throw std::invalid_argument("Quadrature precision is too big");
        return LebedevQuad(*it);
    }
    static LebedevQuadBank &instance() {
        static LebedevQuadBank This;
        return This;
    }
    std::vector<BankEntry> entries;
    LebedevQuadBank();
    LebedevQuadBank(const LebedevQuadBank &) = delete;
    LebedevQuadBank &operator=(const LebedevQuadBank &) = delete;
};

#endif
