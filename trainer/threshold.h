#include<iostream>
#include<vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen/dense>

using namespace Eigen;
using namespace std;
using namespace cv;

#ifndef NUM
#define NUM
const int NUM_F = 100;
const int NUM_NF = 100;
#endif
const int B = 100;

void find_bin(
  vector<vector<int> >& f_val, 
  vector<vector<int> >& nf_val, 
  vector<vector<char> >& f_b, 
  vector<vector<char> >& nf_b)
{
	for(int i = 0; i < NUM_T; i++)
	{
		int f_min = INT_MAX, f_max = INT_MIN;
		for(int j = 0; j < f_val[i].size(); j++)
		{
			if(f_val[i][j] < f_min)
				f_min = f_val[i][j];
			
			if(f_val[i][j] > f_max)
				f_max = f_val[i][j];
		}

		for(int j = 0; j < nf_val[i].size(); j++)
		{
			if(nf_val[i][j] < f_min)
				f_min = nf_val[i][j];

			if(nf_val[i][j] > f_max)
				f_max = nf_val[i][j];
		}

		double step = (f_max - f_min) / double(B);
		for(int j = 0; j < f_val[i].size(); j++)
		{
			int b = (f_val[i][j] - f_min) / step;
			if(b >= B)
				b = B - 1;
			f_b[i][j] = b;
		}

		for(int j = 0; j < nf_val[i].size(); j++)
		{
			int b = (nf_val[i][j] - f_min) / step;
			if(b >= B)
				b = B - 1;
			nf_b[i][j] = b;
		}

		f_val[i].clear();
		nf_val[i].clear();
	}
}

int find_best(
  vector<vector<char> >& f_b, 
  vector<vector<char> >& nf_b, 
  double* D, bool* visited)
{
	int min_ind = -1;
	double min_error = FLT_MAX;
	double pt[B],qt[B];
	for(int i = 0; i < NUM_T; i++)
	{
		memset(pt,0,sizeof(double)*B);
		memset(qt,0,sizeof(double)*B);
		for(int j = 0; j < NUM_F; j++)
		{
			pt[f_b[i][j]] += D[j];
		}
		for(int j = 0; j < NUM_NF; j++)
		{
			qt[nf_b[i][j]] += D[j+NUM_F];
		}

		double error = 0;
		for(int j = 0; j < B; j++)
		{
			error += 2 * sqrt(pt[j] * qt[j]);
		}

		if(error < min_error && !visited[i])
		{
			min_error = error;
			min_ind = i;
		}
	}
	cout<<min_error<<" ";
	return min_ind;
}
