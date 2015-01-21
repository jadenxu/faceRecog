#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include "pre_process.h"
#include "threshold.h"
#include "my_draw.h"

int main()
{
	vector<MatrixXd> i_faces;
	vector<MatrixXd> i_nfaces;
	read_img(i_faces, i_nfaces);
	
	vector<vector<int> > ftr;
	ftr.resize(32384);
	for(int i = 0; i < 32384; i++)
		ftr[i].resize(4);
	find_feature(ftr);

	vector<vector<int> > f_val;
	f_val.resize(NUM_T);
	for(int i = 0; i < NUM_T; i++)
		f_val[i].resize(NUM_F);
	vector<vector<int> > nf_val;
	nf_val.resize(NUM_T);
	for(int i = 0; i < NUM_T; i++)
		nf_val[i].resize(NUM_NF);

	feature_value(i_faces, i_nfaces, ftr, f_val, nf_val);
	i_faces.clear();
	i_nfaces.clear();
	
	int T = 100;
	double* D = new double[NUM_F + NUM_NF];
	for(int i = 0; i < NUM_F; i++)
		D[i] = 0.5 / NUM_F;
	for(int i = 0; i < NUM_NF; i++)
		D[i+NUM_F] = 0.5 / NUM_NF;
	int* po = new int[NUM_T];
	int* th = new int[NUM_T];
	bool* visited = new bool[NUM_T];
	memset(visited, 0, sizeof(bool)*NUM_T);
	vector<weak_cla> ada; 
	ada.resize(T);
	double* error = new double[NUM_T];
	//ofstream output;
	//output.open("error.txt");
	ofstream output;
	
	for(int t = 0; t < T; t++)
	{
		//find the theta and po
		find_threshold(f_val, nf_val, th, po,D, error);
		if(t == 0 || t==10 || t == 50 || t == 99)
		{
			output.open("error" + to_string(t) + ".txt");
			vector<double> tem_v(error, error + NUM_T);
			sort(tem_v.begin(), tem_v.end());
			for(int i = 0; i < 1000; i++)
				output<<tem_v[i]<<" ";
			output.close();
		}
		//calcualte the error
		double min_error = INT_MAX;
		int ht = -1;
		for(int i = 0; i < NUM_T; i++)
		{
			//if(t == 0)
				//output<<i<<" "<<error[i]<<endl;
			if(error[i] < min_error && !visited[i])
			{
				min_error = error[i];
				ht = i;
			}
		}
		visited[ht] = true;
		cout<<min_error<<" ";
		ada[t].alpha = 0.5 * log((1-min_error-1e-9)/(min_error+1e-9));
		ada[t].pol = po[ht];
		ada[t].theta = th[ht];
		ada[t].index = ht;
		//cout<<ht<<endl;

		double sum_d = 0;
		for(int i = 0; i < NUM_F; i++)
		{
			if(po[ht] * (f_val[ht][i] - th[ht]) < 0)
				D[i] = D[i] * exp(ada[t].alpha);
			else
				D[i] = D[i] * exp(-ada[t].alpha);
			sum_d += D[i];
		}
		for(int i = 0; i < NUM_NF; i++)
		{
			if(po[ht] * (nf_val[ht][i] - th[ht]) >= 0)
				D[i+NUM_F] = D[i+NUM_F] * exp(ada[t].alpha);
			else
				D[i+NUM_F] = D[i+NUM_F] * exp(-ada[t].alpha);
			sum_d += D[i+NUM_F];
		}
		//cout<<sum_d<<endl;
		for(int i = 0; i < NUM_F + NUM_NF; i++)
		{
			D[i] = D[i] / sum_d;
		}
		if(t == 10 || t == 50 || t == 99)
		{
			double tem_sum;
			output.open("his_f_"+to_string(t) + ".txt");
			int wf = 0;
			for(int i = 0; i < NUM_F; i++)
			{
				tem_sum = 0;
				for(int j = 0; j <= t; j++)
				{
					tem_sum += ada[j].pol * ada[j].alpha * (f_val[ada[j].index][i] - ada[j].theta >= 0 ? 1 : -1);
				}
				output<<tem_sum<<" ";
				if(tem_sum < 0)
					wf++;
			}
			cout<<wf<<" ";
			output.close();
			output.open("his_nf_"+to_string(t) + ".txt");
			int wnf = 0;
			for(int i = 0; i < NUM_NF; i++)
			{
				tem_sum = 0;
				for(int j = 0; j <= t; j++)
				{
					tem_sum += ada[j].pol * ada[j].alpha * (nf_val[ada[j].index][i] - ada[j].theta >= 0 ? 1 : -1);
				}
				output<<tem_sum<<" ";
				if(tem_sum >= 0)
					wnf++;
			}
			cout<<wnf<<endl;
			output.close();
		}
	}

	output.open("ada.txt");
	for(int i = 0; i < T; i++)
	{
		output<<ada[i].index<<" "<<ada[i].pol<<" "<<ada[i].alpha<<" "<<ada[i].theta<<endl;
	}
	output.close();
	
	/*for(int i = 0; i < T; i++)
	{
		int ind = ada[i].index;
		cout<<ftr[ind][0]<<" "<<ftr[ind][1]<<" "<<ftr[ind][2]<<" "<<ftr[ind][3]<<endl;
	}*/
	draw_feature(ada, ftr);

	return 0;
}
