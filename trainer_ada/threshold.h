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
const int NUM_F = 5000;
const int NUM_NF = 10000;
int NUM_T = 32384;
#endif

class point
{
public:
	point()
	{
		value = 0;
		face = 0;
		weight = 0;
	}
	point(int value, bool face, double weight)
	{
		this->value = value;
		this->face = face;
		this->weight = weight;
	}
	int value;
	bool face;
	double weight;
};

struct cmp
{
	bool operator()(const point& lhs, const point& rhs)
	{
		return lhs.value > rhs.value;
	}
};

void find_threshold(
  vector<vector<int> >& f_val, vector<vector<int> >& nf_val, 
  int* th, int* po, double* D, double* error)
{
	for(int i = 0; i < NUM_T; i++)
	{
		vector<point> val;
		for(int j = 0; j < NUM_F; j++)
		{
			val.push_back(point(f_val[i][j], true, D[j]));
		}
		for(int j = 0; j < NUM_NF; j++)
		{
			val.push_back(point(nf_val[i][j], false, D[j+NUM_F]));
		}
		sort(val.begin(), val.end(), cmp());
		double t_p = 0, t_m = 0;
		double s_p = 0, s_m = 0;
		error[i] = FLT_MAX;
		double sum_p = 0, sum_m = 0;
		for(int j = 0; j < val.size(); j++)
		{
			if(val[j].face)
				t_p += val[j].weight;
			else
				t_m += val[j].weight;
		}	
		
		for(int j = 0; j < val.size(); j++)
		{
			if(val[j].face)
				sum_p += val[j].weight;
			else
				sum_m += val[j].weight;
			if((j != val.size() - 1 && val[j].value != val[j+1].value) || (j == val.size() - 1))
			{
				s_p += sum_p;
				s_m += sum_m;
				sum_p = sum_m = 0;
			}

			double e1 = s_m + (t_p - s_p);
			double e2 = s_p + (t_m - s_m);
			if(min(e1,e2) < error[i])
			{
				error[i] = min(e1,e2);
				th[i] = val[j].value;
				if(e1 < e2)
					po[i] = 1;
				else
					po[i] = -1;

				/*if(error[i] <= 0)
				{
					for(int o = 0; o < val.size(); o++)
						cout<<val[o].value<<" "<<val[o].face<<" ";
					cout<<endl;
				}*/
			}

			/*for(int k = 0 ; k < val.size(); k++)
			{
				cout<<val[k].value;
				if(val[k].face)
					cout<<"+ ";
				else
					cout<<"- ";
			}
			cout<<endl;
			cout<<th[i]<<" "<<po[i]<<" "<<s_p<<" "<<s_m<<" "<<error[i]<<endl;*/
		}
		//cout<<"-------------------------------------"<<endl;
		val.clear();
	}
}
