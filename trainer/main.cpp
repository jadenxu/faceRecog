#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include "pre_process.h"
#include "threshold.h"

int main()
{
	vector<MatrixXd> i_faces;
	vector<MatrixXd> i_nfaces;
	read_img(i_faces, i_nfaces);

	vector<vector<int> > ftr;
	ftr.resize(NUM_T);
	for(int i = 0; i < NUM_T; i++)
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

	vector<vector<char> > f_b;
	f_b.resize(NUM_T);
	for(int i = 0; i < NUM_T; i++)
		f_b[i].resize(NUM_F);
	vector<vector<char> > nf_b;
	nf_b.resize(NUM_T);
	for(int i = 0; i < NUM_T; i++)
		nf_b[i].resize(NUM_NF);
	find_bin(f_val, nf_val, f_b, nf_b);
	f_val.clear();
	nf_val.clear();
	
	double* D = new double[NUM_F + NUM_NF];
	for(int i = 0; i < NUM_F; i++)
		D[i] = 0.5 / NUM_F;
	for(int i = 0; i < NUM_NF; i++)
		D[i+NUM_F] = 0.5 / NUM_NF;

	bool* visited = new bool[NUM_T];
	memset(visited, 0, sizeof(bool)*NUM_T);

	int T = 100;
	vector<vector<double> > real;
	real.resize(T);
	for(int i = 0; i < T; i++)
		real[i].resize(B+1);
	ofstream output;
	
	for(int t = 0; t < T; t++)
	{
		//cout<<t<<endl;
		//find the best
		vector<double> pt, qt;
		pt.resize(B);
		qt.resize(B);
		int ht = find_best(f_b, nf_b, D, visited);
		visited[ht] = true;
		real[t][0] = ht;
		for(int j = 0; j < NUM_F; j++)
		{
			pt[f_b[ht][j]] += D[j];
		}
		for(int j = 0; j < NUM_NF; j++)
		{
			qt[nf_b[ht][j]] += D[j+NUM_F];
		}

		for(int i = 0; i < B; i++)
		{
			if(pt[i] == 0)
				real[t][i+1] = -2;
			else if(qt[i] == 0)
				real[t][i+1] = 2;
			else
				real[t][i+1] = 0.5 * log(pt[i]/qt[i]);
		}

		double sum_d = 0;
		for(int i = 0; i < NUM_F; i++)
		{
			D[i] = D[i] * exp(-real[t][f_b[ht][i]+1]);
			sum_d += D[i];
		}
		for(int i = 0; i < NUM_NF; i++)
		{
			D[i+NUM_F] = D[i+NUM_F] * exp(real[t][nf_b[ht][i]+1]);
			sum_d += D[i+NUM_F];
		}
		for(int i = 0; i < NUM_F + NUM_NF; i++)
		{
			D[i] = D[i] / sum_d;
		}
		qt.clear();
		pt.clear();
		if(t == 10 || t == 50 || t == 99)
		{
			int wf = 0;
			double tem_sum;
			output.open("his_f_"+to_string(t) + ".txt");
			for(int i = 0; i < NUM_F; i++)
			{
				tem_sum = 0;
				for(int j = 0; j <= t; j++)
				{
					int b = f_b[real[j][0]][i];
					tem_sum += real[j][b];
				}
				output<<tem_sum<<" ";
				if(tem_sum < 0)
					wf++;
			}
			cout<<wf<<" ";
			output.close();
			int wnf = 0;
			output.open("his_nf_"+to_string(t) + ".txt");
			for(int i = 0; i < NUM_NF; i++)
			{
				tem_sum = 0;
				for(int j = 0; j <= t; j++)
				{
					int b = nf_b[real[j][0]][i];
					tem_sum += real[j][b];
				}
				output<<tem_sum<<" ";
				if(tem_sum >= 0)
					wnf++;
			}
			cout<<wnf<<endl;
			output.close();
		}
	}
  
  return 0;
}
