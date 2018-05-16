/*
LSTM的基础实现，供和我一样的初学者参考，欢迎交流、共同进步。

author: 大火同学
date:   2018/4/28
email:  12623862@qq.com
*/

#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "lstm.h"

using namespace std;


//激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//激活函数的导数
double dsigmoid(double y){
    return y * (1.0 - y);  
}           

//tanh的导数
double dtanh(double y){
    y = tanh(y);
    return 1.0 - y * y;  
}

/*
权值初始化
参数：
w、二维权值首地址
x、行数
y、列数
*/
void initW(double **w, int x, int y){
    FOR(i, x){
        FOR(j, y)
            w[i][j] = RANDOM_VALUE();  //随机分布   -1~1
    }
}

/*
打印lstm cell单元的状态，用于调试
*/
void Lstm::showStates(){
	FOR(s, _states.size()){
		cout<<"states["<<s<<"]:"<<endl<<"I_G\t\tF_G\t\tO_G\t\tN_I\t\tS\t\tH"<<endl;
		FOR(i, _hideNodeNum){
			cout<<_states[s]->I_G[i]<<"\t\t";
			cout<<_states[s]->F_G[i]<<"\t\t";
			cout<<_states[s]->O_G[i]<<"\t\t";
			cout<<_states[s]->N_I[i]<<"\t\t";
			cout<<_states[s]->S[i]<<"\t\t";
			cout<<_states[s]->H[i]<<"\n";
		}
		cout<<"Y:";
		FOR(i, _outNodeNum){
			cout<<_states[s]->Y[i]<<"\t";
		}
		cout<<endl;
	}
}

/*
清除单元状态
*/
void Lstm::resetStates(){
	FOR(i, _states.size()){
		delete _states[i];
	}
	_states.clear();
}

/*
打印权值，用于调试
*/
void Lstm::showWeights(){
	cout<<"--------------------Wx+b=Y-----------------"<<endl;
	FOR(i, _outNodeNum){
    	cout<<"_W_Y:\n";
    	FOR(j, _hideNodeNum){
    		cout<<_W_Y[j][i]<<"\t";
    	}
    	cout<<"\n_BY:\n"<<_B_Y[i];
    }

    cout<<"\n\n-------------------------Wx+Uh+b=Y----------------------------"<<endl;
    FOR(j, _hideNodeNum){
    	cout<<"\n------------------\nU_:\n";
    	FOR(k, _hideNodeNum){
    		cout<<_U_I[k][j]<<"|"<<_U_F[k][j]<<"|"<<_U_O[k][j]<<"|"<<_U_G[k][j]<<endl;
    	}
    	cout<<"\nW_:\n";
    	FOR(k, _inNodeNum){
    		cout<<_W_I[k][j]<<"|"<<_W_F[k][j]<<"|"<<_W_O[k][j]<<"|"<<_W_G[k][j]<<endl;
    	}

        cout<<"\nB_:\n";
    	cout<<_B_I[j]<<"|"<<_B_F[j]<<"|"<<_B_O[j]<<"|"<<_B_G[j]<<endl;
    }
    cout<<endl<<"---------------------------------------------------"<<endl;
}

/*
初始化网络权值
*/
void Lstm::renewWeights(){
	initW(_W_I, _inNodeNum, _hideNodeNum);
    initW(_U_I, _hideNodeNum, _hideNodeNum);
    initW(_W_F, _inNodeNum, _hideNodeNum);
    initW(_U_F, _hideNodeNum, _hideNodeNum);
    initW(_W_O, _inNodeNum, _hideNodeNum);
    initW(_U_O, _hideNodeNum, _hideNodeNum);
    initW(_W_G, _inNodeNum, _hideNodeNum);
    initW(_U_G, _hideNodeNum, _hideNodeNum);
    initW(_W_Y, _hideNodeNum, _outNodeNum);

    memset(_B_I, 0, sizeof(double)*_hideNodeNum);
    memset(_B_O, 0, sizeof(double)*_hideNodeNum);
    memset(_B_G, 0, sizeof(double)*_hideNodeNum);
    memset(_B_F, 0, sizeof(double)*_hideNodeNum);
    memset(_B_Y, 0, sizeof(double)*_outNodeNum);
}


/*
构造函数
参数：
innode、输入单元个数（特征数）
hidenode、隐藏单元个数
outnode、输出单元个数（结果维度）
*/
Lstm::Lstm(int innode, int hidenode, int outnode){
    _inNodeNum = innode;
    _hideNodeNum = hidenode;
    _outNodeNum = outnode;
	_verification = 0;
	_learningRate = LEARNING_RATE;

    //动态初始化权值
    _W_I = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_F = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_O = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_G = (double**)malloc(sizeof(double*)*_inNodeNum);
    FOR(i, _inNodeNum){
        _W_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _U_I = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_F = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_O = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_G = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _U_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _B_I = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_F = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_O = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_G = (double*)malloc(sizeof(double)*_hideNodeNum);

    _W_Y = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _W_Y[i] = (double*)malloc(sizeof(double)*_outNodeNum);
    }
    _B_Y = (double*)malloc(sizeof(double)*_outNodeNum);

	renewWeights();

    cout<<"Lstm instance inited."<<endl;
}

/*
析构函数，释放内存
*/
Lstm::~Lstm(){
	resetStates();

    FOR(i, _inNodeNum){
        if(_W_I[i]!=NULL){
            free(_W_I[i]);
            _W_I[i]=NULL;
        }
        if(_W_F[i]!=NULL){
            free(_W_F[i]);
            _W_F[i]=NULL;
        }
        if(_W_O[i]!=NULL){
            free(_W_O[i]);
            _W_O[i]=NULL;
        }
        if(_W_G[i]!=NULL){
            free(_W_G[i]);
            _W_G[i]=NULL;
        }
    }
    if(_W_I!=NULL){
        free(_W_I);
        _W_I=NULL;
    }
    if(_W_F!=NULL){
        free(_W_F);
        _W_F=NULL;
    }
    if(_W_O!=NULL){
        free(_W_O);
        _W_O=NULL;
    }
    if(_W_G!=NULL){
        free(_W_G);
        _W_G=NULL;
    }

    FOR(i, _hideNodeNum){
        if(_U_I[i]!=NULL){
            free(_U_I[i]);
            _U_I[i]=NULL;
        }
        if(_U_F[i]!=NULL){
            free(_U_F[i]);
            _U_F[i]=NULL;
        }
        if(_U_O[i]!=NULL){
            free(_U_O[i]);
            _U_O[i]=NULL;
        }
        if(_U_G[i]!=NULL){
            free(_U_G[i]);
            _U_G[i]=NULL;
        }
    }
    if(_U_I!=NULL){
        free(_U_I);
        _U_I=NULL;
    }
    if(_U_F!=NULL){
        free(_U_F);
        _U_F=NULL;
    }
    if(_U_O!=NULL){
        free(_U_O);
        _U_O=NULL;
    }
    if(_U_G!=NULL){
        free(_U_G);
        _U_G=NULL;
    }


    if(_B_I!=NULL){
        free(_B_I);
        _B_I=NULL;
    }
    if(_B_F!=NULL){
        free(_B_F);
        _B_F=NULL;
    }
    if(_B_O!=NULL){
        free(_B_O);
        _B_O=NULL;
    }
    if(_B_G!=NULL){
        free(_B_G);
        _B_G=NULL;
    }


    FOR(i, _hideNodeNum){
        if(_W_Y[i]!=NULL){
            free(_W_Y[i]);
            _W_Y[i]=NULL;
        }
    }
    if(_W_Y!=NULL){
        free(_W_Y);
        _W_Y=NULL;
    }
    if(_B_Y!=NULL){
        free(_B_Y);
        _B_Y=NULL;
    }

    cout<<"Lstm instance has been destroyed."<<endl;
}

/*
计算训练集的损失
参数：
x、训练特征集
y、训练标签集
*/
double Lstm::trainLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	len -= _verification*len;//训练集长度
	FOR(i, len){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
		// delete state;
		// state = NULL;
        _states.push_back(state);
	}
	rmse = error/(len*_outNodeNum);
	return rmse;
}


/*
计算验证集的损失，参数于上一个函数相同，通过_verification计算验证集的起始下标
参数：
x、训练特征集
y、训练标签集
*/
double Lstm::verificationLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	int start = len-_verification*len;//验证集起始下标
	if(start==len) return 0;//验证集数量为0
	for(int i=start;i<len;++i){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
        // delete state;
        // state = NULL;
        _states.push_back(state);
	}
	rmse = error/((len-start)*_outNodeNum);
	return rmse;
}


/*
单个样本正向传播
参数：
x、单个样本特征向量
*/
LstmStates *Lstm::forward(DataType *x){
	if(x==NULL){
		return 0;
	}

    LstmStates *lstates = new LstmStates(_hideNodeNum, _outNodeNum);
 //    LstmStates *lstates = (LstmStates*)malloc(sizeof(LstmStates));
	// memset(lstates, 0, sizeof(LstmStates));

	//上个时间点的状态
	if(_states.size()>0){
		memcpy(lstates->PreS, _states[_states.size()-1]->S, sizeof(double)*_hideNodeNum);
		memcpy(lstates->PreH, _states[_states.size()-1]->H, sizeof(double)*_hideNodeNum);
	}

    //输入层转播到隐层
    FOR(j, _hideNodeNum){   
        double inGate = 0.0;
        double outGate = 0.0;
        double forgetGate = 0.0;
        double newIn = 0.0;
        // double s = 0.0;

        FOR(m, _inNodeNum){
            inGate += x[m] * _W_I[m][j]; 
            outGate += x[m] * _W_O[m][j];
            forgetGate += x[m] * _W_F[m][j];
            newIn += x[m] * _W_G[m][j];
        }

        FOR(m, _hideNodeNum){
            inGate += lstates->PreH[m] * _U_I[m][j];
            outGate += lstates->PreH[m] * _U_O[m][j];
            forgetGate += lstates->PreH[m] * _U_F[m][j];
            newIn += lstates->PreH[m] * _U_G[m][j];
        }

        inGate += _B_I[j];
        outGate += _B_O[j];
        forgetGate += _B_F[j];
        newIn += _B_G[j];

        lstates->I_G[j] = sigmoid(inGate);   
        lstates->O_G[j] = sigmoid(outGate);
        lstates->F_G[j] = sigmoid(forgetGate);
        lstates->N_I[j] = tanh(newIn);

        //得出本时间点状态
        lstates->S[j] = lstates->F_G[j]*lstates->PreS[j]+(lstates->N_I[j]*lstates->I_G[j]);
        //本时间点的输出
        // lstates->H[j] = lstates->I_G[j]*tanh(lstates->S[j]);//!!!!!!
        lstates->H[j] = lstates->O_G[j]*tanh(lstates->S[j]);//changed
    }


    //隐藏层传播到输出层
    double out = 0.0;
    FOR(i, _outNodeNum){
	    FOR(j, _hideNodeNum){
	        out += lstates->H[j] * _W_Y[j][i];
	    }
	    out += _B_Y[i];
	    // lstates->Y[i] = sigmoid(out);//输出层各单元输出
	    lstates->Y[i] = out;//输出层各单元输出
	}

    return lstates;
}

/*
正向传播，暂未实现按batch_size计算。
参数：
trainSet、训练特征集，vector<特征向量（向量长度需与输入单元数量相同）>
labelSet、训练标签集，vector<标签向量（向量长度需与输出单元数量相同）>
*/
void Lstm::forward(vector<DataType*> trainSet, vector<DataType*> labelSet){
	int len = trainSet.size();
	len -= _verification*len;//减去验证集
	FOR(i, len){
		LstmStates *state = forward(trainSet[i]);
	    //保存标准误差关于输出层的偏导
	    double delta = 0.0;
	    FOR(j, _outNodeNum){//
	    	// delta = (labelSet[i][j]-state->Y[j])*dsigmoid(state->Y[j]);//!!!!!!!!!
	    	// delta = (labelSet[i][j]-state->Y[j]);//changed
	    	delta = 2*(state->Y[j]-labelSet[i][j]);//loss=label^2-2*label*y+y^2;   dloss/dy=2y-2label; 
	    	state->yDelta[j] = delta;
	    }
	    _states.push_back(state);
	}
}

/*
反向传播,计算各个权重的偏导数
参数：
trainSet、训练特征集，vector<特征向量（向量长度需与输入单元数量相同）>
deltas、存储每个权值偏导数的对象指针
*/
void Lstm::backward(vector<DataType*> trainSet, Deltas *deltas){
	if(_states.size()<=0){
		cout<<"need go forward first."<<endl;
	}
    //隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
    double hDelta[_hideNodeNum];  
    double *oDelta = new double[_hideNodeNum];
    double *iDelta = new double[_hideNodeNum];
    double *fDelta = new double[_hideNodeNum];
    double *nDelta = new double[_hideNodeNum];
    double *sDelta = new double[_hideNodeNum];

    //当前时间之后的一个隐藏层误差
    double *oPreDelta = new double[_hideNodeNum]; 
    double *iPreDelta = new double[_hideNodeNum];
    double *fPreDelta = new double[_hideNodeNum];
    double *nPreDelta = new double[_hideNodeNum];
    double *sPreDelta = new double[_hideNodeNum];
    double *fPreGate = new double[_hideNodeNum];

    memset(oPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(iPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(fPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(nPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(sPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(fPreGate, 0, sizeof(double)*_hideNodeNum);


    int p = _states.size()-1;
    for(; p>=0; --p){//batch=1
    	// cout<<"p="<<p<<"|size:"<<_states.size()<<endl;
        //当前隐藏层
        double *inGate = _states[p]->I_G;     //输入门
        double *outGate = _states[p]->O_G;    //输出门
        double *forgetGate = _states[p]->F_G; //遗忘门
        double *newInGate = _states[p]->N_I;  //新记忆
        double *state = _states[p]->S;     //状态值
        double *h = _states[p]->H;         //隐层输出值

        //前一个隐藏层
        double *preH = _states[p]->PreH;   
        double *preState = _states[p]->PreS;

        FOR(k, _outNodeNum){  //对于网络中每个输出单元，更新权值
            //更新隐含层和输出层之间的连接权
            FOR(j, _hideNodeNum){
                deltas->dwy[j][k].data += _states[p]->yDelta[k] * h[j];
                // _W_Y[j][k] -= _learningRate * _states[p]->yDelta[k] * h[j];
            }
            deltas->dby[k].data += _states[p]->yDelta[k];
            // _B_Y[k] -= _learningRate * _states[p]->yDelta[k];
        }

        //目标函数对于网络中每个隐藏单元的偏导数计算
        FOR(j, _hideNodeNum){
            //隐含层的各个门及单元状态
            oDelta[j] = 0.0;
            iDelta[j] = 0.0;
            fDelta[j] = 0.0;
            nDelta[j] = 0.0;
            sDelta[j] = 0.0;
            hDelta[j] = 0.0;

            //目标函数对隐藏状态的偏导数
            FOR(k, _outNodeNum){
                hDelta[j] += _states[p]->yDelta[k] * _W_Y[j][k];
            }
            FOR(k, _hideNodeNum){
                hDelta[j] += iPreDelta[k] * _U_I[j][k];
                hDelta[j] += fPreDelta[k] * _U_F[j][k];
                hDelta[j] += oPreDelta[k] * _U_O[j][k];
                hDelta[j] += nPreDelta[k] * _U_G[j][k];
            }

            oDelta[j] = hDelta[j] * tanh(state[j]) * dsigmoid(outGate[j]);
            sDelta[j] = hDelta[j] * outGate[j] * dtanh(state[j]) + sPreDelta[j] * fPreGate[j];
            fDelta[j] = sDelta[j] * preState[j] * dsigmoid(forgetGate[j]);
            iDelta[j] = sDelta[j] * newInGate[j] * dsigmoid(inGate[j]);
            nDelta[j] = sDelta[j] * inGate[j] * dtanh(newInGate[j]);

            //更新前一个隐含层和现在隐含层之间的权值
            FOR(k, _hideNodeNum){
                deltas->dui[k][j].data += iDelta[j] * preH[k];
                deltas->duf[k][j].data += fDelta[j] * preH[k];
                deltas->duo[k][j].data += oDelta[j] * preH[k];
                deltas->dun[k][j].data += nDelta[j] * preH[k];
            }

            //更新输入层和隐含层之间的连接权
            FOR(k, _inNodeNum){
                deltas->dwi[k][j].data += iDelta[j] * trainSet[p][k];
                deltas->dwi[k][j].data += fDelta[j] * trainSet[p][k];
                deltas->dwo[k][j].data += oDelta[j] * trainSet[p][k];
                deltas->dwn[k][j].data += nDelta[j] * trainSet[p][k];
            }

            deltas->dbi[j].data += iDelta[j];
            deltas->dbf[j].data += fDelta[j];
            deltas->dbo[j].data += oDelta[j];
            deltas->dbn[j].data += nDelta[j];
        }

        if(p == (_states.size()-1)){
            delete  []oPreDelta;
            delete  []fPreDelta;
            delete  []iPreDelta;
            delete  []nPreDelta;
            delete  []sPreDelta;
            delete  []fPreGate;
        }

        oPreDelta = oDelta;
        fPreDelta = fDelta;
        iPreDelta = iDelta;
        nPreDelta = nDelta;
        sPreDelta = sDelta;
        fPreGate = forgetGate;
	}
    delete  []oPreDelta;
    delete  []fPreDelta;
    delete  []iPreDelta;
    delete  []nPreDelta;
    delete  []sPreDelta;

	return;
}

/*
根据各权值的偏导数更新权值
参数：
deltaSet、存储每个权值偏导数的对象指针
epoche、当前迭代次数
*/
void Lstm::optimize(Deltas *deltaSet, int epoche){
    FOR(i, _outNodeNum){
    	FOR(j, _hideNodeNum){
    		_W_Y[j][i] = deltaSet->dwy[j][i].optimize(_W_Y[j][i], epoche);
    	}
    	_B_Y[i] = deltaSet->dby[i].optimize(_B_Y[i], epoche);
    }

    FOR(j, _hideNodeNum){
    	FOR(k, _hideNodeNum){
    		_U_I[k][j] = deltaSet->dui[k][j].optimize(_U_I[k][j], epoche);
    		_U_F[k][j] = deltaSet->duf[k][j].optimize(_U_F[k][j], epoche);
    		_U_O[k][j] = deltaSet->duo[k][j].optimize(_U_O[k][j], epoche);
    		_U_G[k][j] = deltaSet->dun[k][j].optimize(_U_G[k][j], epoche);
    	}
    	FOR(k, _inNodeNum){
    		_W_I[k][j] = deltaSet->dwi[k][j].optimize(_W_I[k][j], epoche);
    		_W_F[k][j] = deltaSet->dwf[k][j].optimize(_W_F[k][j], epoche);
    		_W_O[k][j] = deltaSet->dwo[k][j].optimize(_W_O[k][j], epoche);
    		_W_G[k][j] = deltaSet->dwn[k][j].optimize(_W_G[k][j], epoche);
    	}

        _B_I[j] = deltaSet->dbi[j].optimize(_B_I[j], epoche);
        _B_F[j] = deltaSet->dbf[j].optimize(_B_F[j], epoche);
        _B_O[j] = deltaSet->dbo[j].optimize(_B_O[j], epoche);
        _B_G[j] = deltaSet->dbn[j].optimize(_B_G[j], epoche);
    }
}

double _LEARNING_RATE = LEARNING_RATE;//用于sgd优化器的全局学习率
/*
训练网络
参数：
trainSet、训练特征集
labelSet、训练标签集
epoche、迭代次数
verification、验证集的比例
stopThreshold、提前停止阈值，当两次迭代结果的变化小于此阈值则停止
*/
void Lstm::train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold){
	if(trainSet.size()<=0 || labelSet.size()<=0 || trainSet.size()!=labelSet.size()){
		cout<<"data set error!"<<endl;
		return;
	}


    _verification = 0;
    if(verification>0 && verification<0.5){
        _verification = verification;
    }else{
        cout<<"verification rate is invalid."<<endl;
    }

	double lastTrainRmse = 0.0;
	double lastVerRmse = 0.0;
    _LEARNING_RATE = LEARNING_RATE;//开始训练前初始化学习率 适用于SGD优化器

    //计算验证集的平均值
    double verificationAvg = 0.0;
    if(_verification>0){
        int verLen = _verification*labelSet.size();
        FOR(i, verLen){
        	verificationAvg += labelSet[labelSet.size()-verLen+i][0];
        }
        verificationAvg /= verLen;
        verificationAvg = verificationAvg<0?-verificationAvg:verificationAvg;
        cout<<"---------------avg="<<verificationAvg<<endl;
    }

    Deltas *deltaSet = new Deltas(_inNodeNum, _hideNodeNum, _outNodeNum);
    cout<<"deltaset inited. start trainning."<<endl;
	FOR(e, epoche){	
		//每次epoche清除单元状态
		resetStates();
		//正向传播
		forward(trainSet, labelSet);
		//反向计算误差并计算偏导数
        deltaSet->resetDelta();//重置每个权值的偏导数值
		backward(trainSet, deltaSet);
		//根据偏导数更新权重
		optimize(deltaSet, e);

		//验证前清除单元状态
		resetStates();
		double trainRmse = trainLoss(trainSet, labelSet);
		double verRmse = verificationLoss(trainSet, labelSet);
		// cout<<"epoche:"<<e<<"|rmse:"<<trainRmse<<endl;
		if(e>0 && abs(trainRmse-lastTrainRmse) < stopThreshold){//变化足够小
			cout<<"train rmse got tiny diff, stop in epoche:"<<e<<endl;
			break;
		}

		if(e>0 && verRmse!=0 && (verRmse-lastVerRmse)>(verificationAvg*0.025)){//验证集准确率大幅下降则停止 0.03~0.04(84.792)
			// cout<<"verification rmse ascend too much:"<<verRmse-lastVerRmse<<", stop in epoche:"<<e<<endl;
			// cout<<"verification rmse ascend or got tiny diff, stop in epoche:"<<e<<endl;
			break;
		}

		lastTrainRmse = trainRmse;
		lastVerRmse = verRmse;
	}
    deltaSet->~Deltas();
    deltaSet = NULL;
}

/*
预测单个样本
参数：
x、需预测样本的特征集
*/
DataType *Lstm::predict(DataType *x){
    // cout<<"predict X>"<<endl;
    // FOR(i, _inNodeNum) cout<<x[i]<<",";
    // cout<<endl;

	LstmStates *state = forward(x);
	DataType *ret = new DataType[_outNodeNum];
	memcpy(ret, state->Y, sizeof(DataType)*_outNodeNum);//备份结果
	// free(state);
	_states.push_back(state);//记住当前时间点的单元状态
    // cout<<"Y>";
    // FOR(i, _outNodeNum) cout<<ret[i]<<",";
    // cout<<endl;
	return ret;
}



//adam优化器
double Optimizer::adam(double preTheta, const double dt, const int time){
	mt = beta1*mt+(1-beta1)*dt;
	vt = beta2*vt+(1-beta2)*(dt*dt);
	double mcap = mt/(1-pow(beta1, time));
	double vcap = vt/(1-pow(beta2, time));
	double theta = preTheta - (lr*mcap)/(sqrt(vcap)+epsilon);

	// cout<<"Adam-preTheta="<<preTheta<<"|mt="<<mt<<"|vt="<<vt<<"|mcap="<<mcap<<"|vcap="<<vcap<<"|time="<<time<<"|theta="<<theta<<endl;
	return theta;
}

//sgd优化器
double Optimizer::sgd(double preTheta, const double dt){
	double theta = preTheta - _LEARNING_RATE*dt;
	return theta;
}

//初始化偏导集合
Deltas::Deltas(const int in, const int hide, const int out){
    _inNodeNum = in;
    _outNodeNum = out;
    _hideNodeNum = hide;

    dwi = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwf = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwo = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwn = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    FOR(i, _inNodeNum){
        dwi[i] = new Delta[_hideNodeNum];
        dwf[i] = new Delta[_hideNodeNum];
        dwo[i] = new Delta[_hideNodeNum];
        dwn[i] = new Delta[_hideNodeNum];
    }

    dui = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duf = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duo = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    dun = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dui[i] = new Delta[_hideNodeNum];
        duf[i] = new Delta[_hideNodeNum];
        duo[i] = new Delta[_hideNodeNum];
        dun[i] = new Delta[_hideNodeNum];
    }

    dbi = new Delta[_hideNodeNum];
    dbf = new Delta[_hideNodeNum];
    dbo = new Delta[_hideNodeNum];
    dbn = new Delta[_hideNodeNum];

    dwy = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dwy[i] = new Delta[_outNodeNum];
    }

    dby = new Delta[_outNodeNum];

}

Deltas::~Deltas(){
    FOR(i, _inNodeNum){
        delete [] dwi[i];
        delete [] dwf[i];
        delete [] dwo[i];
        delete [] dwn[i];
    }
    free(dwi);
    free(dwf);
    free(dwo);
    free(dwn);

    FOR(i, _hideNodeNum){
        delete [] dui[i];
        delete [] duf[i];
        delete [] duo[i];
        delete [] dun[i];
    }
    free(dui);
    free(duf);
    free(duo);
    free(dun);

    FOR(i, _hideNodeNum){
        delete [] dwy[i];
    }
    free(dwy);

    delete [] dbi;
    delete [] dbf;
    delete [] dbo;
    delete [] dbn;
    delete [] dby;
}




LstmStates::LstmStates(const int hide, const int out){
    // std::cout<<"new LstmStates"<<std::endl;
    I_G = (double*)malloc(sizeof(double)*hide);
    F_G = (double*)malloc(sizeof(double)*hide);
    O_G = (double*)malloc(sizeof(double)*hide);
    N_I = (double*)malloc(sizeof(double)*hide);
    S = (double*)malloc(sizeof(double)*hide);
    H = (double*)malloc(sizeof(double)*hide);
    PreS = (double*)malloc(sizeof(double)*hide);
    PreH = (double*)malloc(sizeof(double)*hide);
    Y = (DataType*)malloc(sizeof(DataType)*out);
    yDelta = (double*)malloc(sizeof(double)*out);

    memset(I_G, 0, sizeof(double)*hide);
    memset(F_G, 0, sizeof(double)*hide);
    memset(O_G, 0, sizeof(double)*hide);
    memset(N_I, 0, sizeof(double)*hide);
    memset(S, 0, sizeof(double)*hide);
    memset(H, 0, sizeof(double)*hide);
    memset(PreS, 0, sizeof(double)*hide);
    memset(PreH, 0, sizeof(double)*hide);
    memset(Y, 0, sizeof(DataType)*out);
    memset(yDelta, 0, sizeof(double)*out);
}

LstmStates::~LstmStates(){
    // std::cout<<"delete LstmStates"<<std::endl;
    free(I_G);
    free(F_G);
    free(O_G);
    free(N_I);
    free(S);
    free(H);
    free(PreS);
    free(PreH);
    free(Y);
    free(yDelta);
}


Delta::Delta(){
    opt = new Optimizer();
    data = 0;
}

Delta::~Delta(){
    delete opt;
}

double Delta::optimize(double theta, const int time){
    if(opt!=NULL){
        theta = opt->adam(theta, data, time+1);//time从1开始
        // theta = opt->sgd(theta, data);//time从1开始
    }else{
        theta -= LEARNING_RATE * data;
    }

    return theta;
}

//重置偏导，保存优化器参数状态
void Deltas::resetDelta(){
    FOR(i, _inNodeNum){
        FOR(j, _hideNodeNum){
            dwi[i][j].data = 0;
            dwf[i][j].data = 0;
            dwo[i][j].data = 0;
            dwn[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _hideNodeNum){
            dui[i][j].data = 0;
            duf[i][j].data = 0;
            duo[i][j].data = 0;
            dun[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _outNodeNum){
            dwy[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        dbi[i].data = 0;
        dbf[i].data = 0;
        dbo[i].data = 0;
        dbn[i].data = 0;
    }

    FOR(i, _outNodeNum){
        dby[i].data = 0;
    }
}


