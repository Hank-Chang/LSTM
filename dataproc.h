#ifndef __DATA_PROC_H__
#define __DATA_PROC_H__

#define FOR(i,N) for(int i=0;i<N;++i)

double scale(double *data, const int len);
void invertScale(double *data, const int len, const double scaleRate);

#endif//__DATA_PROC_H__