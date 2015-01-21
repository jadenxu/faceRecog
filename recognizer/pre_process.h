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
int NUM_T = 32384;
#endif

class point
{
public:
	point()
	{
		x = y = num = 0;
	}
	point(int x, int y, int num)
	{
		this->x = x;
		this->y = y;
		this->num = num;
	}
	int x;
	int y;
	int num;
};

struct weak_cla
{
	int index;
	double alpha; double theta; int pol;
};

void find_feature(vector<vector<int> >& ftr)
{
	int count = 0;
	for(int i = 0; i < 17; i++)
	{
		for(int j = i+1; j < 17; j++)
		{
			for(int p = 0; p < 17; p++)
			{
				for(int q = p+2; q < 17; q+=2)
				{
					ftr[count][0] = i; ftr[count][1] = j; ftr[count][2] = p; ftr[count][3] = q;
					ftr[count+8704][0] = p; ftr[count+8704][1] = q; ftr[count+8704][2] = i; ftr[count+8704][3] = j;
					count++;
				}
			}
		}
	}
	count += 8704;
	
	for(int i = 0; i < 17; i++)
	{
		for(int j = i+1; j < 17; j++)
		{
			for(int p = 0; p < 17; p++)
			{
				for(int q = p+3; q < 17; q+=3)
				{
					ftr[count][0] = i; ftr[count][1] = j; ftr[count][2] = p; ftr[count][3] = q;
					ftr[count+5440][0] = p; ftr[count+5440][1] = q; ftr[count+5440][2] = i; ftr[count+5440][3] = j;
					count++;
				}
			}
		}
	}
	count += 5440;

	for(int i = 0; i < 17; i++)
	{
		for(int j = i+2; j < 17; j+=2)
		{
			for(int p = 0; p < 17; p++)
			{
				for(int q = p+2; q < 17; q+=2)
				{
					ftr[count][0] = i; ftr[count][1] = j; ftr[count][2] = p; ftr[count][3] = q;
					count++;
				}
			}
		}
	}
}

int cal_sq(int i, int j, int p, int q, MatrixXd& i_face)
{
	int x = i_face(j,q) + i_face(i,p) - i_face(i,q) - i_face(j,p);
	return x;
}

int feature_value(int num, Mat& img, vector<vector<int> >& ftr)
{
	MatrixXd s_img = MatrixXd::Zero(17,17);
	for(int i = 1; i <= 16; i++)
	{
		for(int j = 1; j <= 16; j++)
		{
			s_img(i,j) = s_img(i,j-1) + img.at<uchar>(i-1,j-1);
		}
	}
	
	MatrixXd i_img = MatrixXd::Zero(17,17);
	for(int i = 1; i <= 16; i++)
	{
		for(int j = 1; j <= 16; j++)
		{
			i_img(i,j) = i_img(i-1,j) + s_img(i,j);
		}
	}

	int h_n[5] = {8704, 17408, 22848, 28288, 32384}; // 2h, 2v, 3h, 3v, 4

	int i = ftr[num][0], j = ftr[num][1], p = ftr[num][2], q = ftr[num][3];
	int value;
	if(num < h_n[0])
		value = cal_sq(i,j,p,(p+q)/2, i_img) - cal_sq(i,j,(p+q)/2,q, i_img);
	else if(num < h_n[1])
		value = cal_sq(i,(i+j)/2,p,q, i_img) - cal_sq((i+j)/2,j,p,q, i_img);
	else if(num < h_n[2])
		value = cal_sq(i,j,p,(p*2+q)/3, i_img) + cal_sq(i,j,(p+q*2)/3,q, i_img) - cal_sq(i,j,(p*2+q)/3,(p+q*2)/3, i_img);
	else if(num < h_n[3])
		value = cal_sq(i,(i*2+j)/3,p,q,i_img) + cal_sq((i+j*2)/3,j,p,q,i_img) - cal_sq((i*2+j)/3,(i+j*2)/3,p,q,i_img);
	else
		value = cal_sq(i,(i+j)/2,p,(p+q)/2,i_img) + cal_sq((i+j)/2,j,(p+q)/2,q, i_img) - cal_sq(i,(i+j)/2,(p+q)/2,q,i_img) - cal_sq((i+j)/2,j,p,(p+q)/2,i_img);

	return value;
}
