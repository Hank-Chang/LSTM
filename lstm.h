/*
LSTM的基础实现，供和我一样的初学者参考，欢迎交流、共同进步。

author: 大火同学
date:   2018/4/28
email:  12623862@qq.com
*/

#ifndef __H_LSTM_H__
#define __H_LSTM_H__


using namespace std;

#define LEARNING_RATE	0.0001
#define RANDOM_VALUE() ((double)rand()/RAND_MAX*2-1)	//-1~1随机
#define FOR(i,N) for(int i=0;i<N;++i)

typedef double DataType;

class LstmStates{

public:
    double *I_G;       //输入门
    double *F_G;       //遗忘门
    double *O_G;       //输出门
    double *N_I;       //新输入
    double *S;     //状态值
    double *H;     //隐层输出值
    DataType *Y;        //输出值
    double *yDelta; //保存误差关于输出层的偏导

    double *PreS;      //上一时刻的状态值
    double *PreH;      //上一时刻的隐层输出值

    LstmStates(const int hide, const int out);
    ~LstmStates();
};

class Optimizer{
private:
	double lr;
	double beta1;
	double beta2;
	double epsilon;
	double mt;
	double vt;
public:
	Optimizer(){
		//adam相关参数
		lr = 0.01;
		beta1 = 0.9;
		beta2 = 0.99;
		epsilon = 1e-8;
		mt = 0.0;
		vt = 0.0;
	};
	~Optimizer(){};
	double adam(double theta, const double dt, const int time);
	double sgd(double theta, const double dt);
};

class Delta{
	Optimizer *opt;
public:
    double data;
    Delta();
    ~Delta();
    double optimize(double theta, const int time);
};

class Deltas{
private:
    int _inNodeNum,_hideNodeNum,_outNodeNum;
public:
    //偏导矩阵，存放每个权值的偏导用于更新权值
    Delta **dwi;
    Delta **dui;
    Delta *dbi;
    Delta **dwf;
    Delta **duf;
    Delta *dbf;
    Delta **dwo;
    Delta **duo;
    Delta *dbo;
    Delta **dwn;
    Delta **dun;
    Delta *dbn;
    Delta **dwy;
    Delta *dby;

    Deltas(int in, int hide, int out);
    ~Deltas();
    void resetDelta();
};

class Lstm{
public:
	Lstm(int innode, int hidenode, int outnode);
	~Lstm();
    void train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold);
	DataType *predict(DataType *X);
	void showStates();
	void showWeights();

private:
	LstmStates *forward(DataType *x);//单个样本正向传播
	void forward(vector<DataType*> trainSet, vector<DataType*> labelSet);//所有样本正向传播
	void backward(vector<DataType*> trainSet, Deltas *deltaSet);//反向更新
	void optimize(Deltas *deltaSet, int epoche);//权重更新
	double trainLoss(vector<DataType*> x, vector<DataType*> y);//使用rmse均方差作为损失函数
	double verificationLoss(vector<DataType*> x, vector<DataType*> y);
	void resetStates();
	void renewWeights();

    int _inNodeNum;
    int _hideNodeNum;
    int _outNodeNum;
	float _verification;						//验证集比例
	vector<LstmStates*> _states;				//隐层单元状态
	double _learningRate;						//学习率

    double **_W_I;    //连接输入与隐含层单元中输入门的权值矩阵
    double **_U_I;  //连接上一隐层输出与本隐含层单元中输入门的权值矩阵
    double *_B_I;
    double **_W_F;    //连接输入与隐含层单元中遗忘门的权值矩阵
    double **_U_F;  //连接上一隐含层与本隐含层单元中遗忘门的权值矩阵
    double *_B_F;
    double **_W_O;    //连接输入与隐含层单元中输出门的权值矩阵
    double **_U_O;  //连接上一隐含层与现在时刻的隐含层的权值矩阵
    double *_B_O;
    double **_W_G;    //连接输入与隐含层单元产生新记忆的权值矩阵
    double **_U_G;  //连接隐含层间单元产生新记忆的权值矩阵
    double *_B_G;
    double **_W_Y;   //连接隐层与输出层的权值矩阵
    double *_B_Y;
};



#endif//__H_LSTM_H__

