#ifndef __CONFIG_H__
#define __CONFIG_H__

/* double is almost useless */
typedef float real;

#define NFREQ (16)

struct MeshElement {
/* 0  */   real  kappa0;
/* 4  */   real  Ip0;
/* 8  */   real  v[3];
/* 20 */   real  Teff;
/* 24 */   real  Te;
/* 28 */   real  dvstep; // ~ 6 / NFREQ
/* 32 */   int   p[4];
/* 48 */   int   neib[4];
};

/*
 * CUDA block size is NFREQ x PTSPERBLOCK
 * Keep block size around 1024
 * */
#define PTSPERBLOCK (1024 / NFREQ)

#endif
