/*
本案例只为说明本lstm用法，数据没有做任何预处理，学习效果并不是很好，
lstm能保留之前时刻的状态，主要是应用于基于时间序列的处理。
而这个案例的数据集完全随机，不存在时间相关性。

本案例使用lstm网络来学习z=x^2-xy+y^2这一函数，x、y作为训练特征，计算出来的z作为训练标签调用train函数进行训练。
训练完成后使用predict来检验学习情况。

author:	大火同学
date:	2018/5/15
email:	12623862@qq.com
*/

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include "lstm.h"

using namespace std;

#define INPUT 	2 
#define HIDE	4
#define OUTPUT 	1

/*
z=x^2-xy+y^2;
*/
double test_function(double x, double y){
	double z = x*x-x*y+y*y;
	return z;
}

int main(){
	vector<double *> trainSet;
	vector<double *> labelSet;

	//固定随机种子，保证每次结果相同
	unsigned long seed = 12345678;
	srand(seed);

	//随机生成1000组训练数据
	FOR(i, 1000){
		double *input=(double*)malloc(sizeof(double)*INPUT);
		double *label=(double*)malloc(sizeof(double)*OUTPUT);
		double x=RANDOM_VALUE();
		double y=RANDOM_VALUE();
		double z=test_function(x,y);
		input[0]=x;
		input[1]=y;
		label[0]=z;
		trainSet.push_back(input);
		labelSet.push_back(label);
	}

	//初始化
	Lstm *lstm = new Lstm(INPUT,HIDE,OUTPUT);

	//投入训练
	cout<<"/*** Learnning function: z=x^2-xy+y^2 ***/"<<endl;
	lstm->train(trainSet, labelSet, 1000, 0, 0.000001);

	//随机生成10组测试数据并对比真实函数结果和lstm所计算的结果。
	FOR(i, 10){
		double *test=(double*)malloc(sizeof(double)*INPUT);
		test[0]=RANDOM_VALUE();
		test[1]=RANDOM_VALUE();
		double *z=lstm->predict(test);
		double rz=test_function(test[0],test[1]);
		double diff=abs(z[0]-rz);
		cout<<"test "<<i<<" x="<<test[0]<<",y="<<test[1]<<", predict z="<<z[0]<<",real z="<<rz<<",deviation＝"<<diff<<endl;
		free(test);
		free(z);
	}

	lstm->~Lstm();

	FOR(i, trainSet.size()){
		free(trainSet[i]);
		free(labelSet[i]);
	}
	trainSet.clear();
	labelSet.clear();
	return 0;
}




