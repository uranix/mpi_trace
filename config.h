#ifndef __CONFIG_H__
#define __CONFIG_H__

/* double is almost useless */
typedef float real;

/*
 * NFREQ % 4 shoud be 0
 * */
#define NFREQ (4)

struct MeshElement {
    real  kappa[NFREQ];
    real  Ip[NFREQ];
    int   p[4];
    int   neib[4];
    char  _padd[
        (NFREQ * sizeof(real) - 1) -
        (2 * NFREQ * sizeof(real) + 8 * sizeof(int) - 1) % (NFREQ * sizeof(real))
    ]; /* may be zero. GCC allows that */
};

/*
 * CUDA block size is NFREQ x PTSPERBLOCK
 * Keep block size around 1024
 * */
#define PTSPERBLOCK (1024 / NFREQ)

#endif
