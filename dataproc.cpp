#include "dataproc.h"

using namespace std;


//缩放
double scale(double *data, const int len){
	double max=0;
	FOR(i,len){
		double tmp = data[i]>0?data[i]:-data[i];
		if(max < tmp){
			max = tmp;
		}
	}

	double scaleRate = max;
	FOR(i,len){
		data[i] /= scaleRate;
	}

	return scaleRate;
}

//逆缩放
void invertScale(double *data, const int len, double scaleRate){
	FOR(i,len){
		data[i] *= scaleRate;
	}
}