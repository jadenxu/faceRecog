#include<iostream>
#include<vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen/dense>

using namespace Eigen;
using namespace std;
using namespace cv;

struct weak_cla
{
	int index;
	double alpha; double theta; int pol;
};

void draw_sq(int i, int j, int p, int q, Mat& img, int color)
{
	for(int row = i; row < j; row++)
	{
		for(int col = p; col < q; col++)
		{
			img.at<Vec3b>(row,col) = Vec3b(255,255,255) * color;
		}
	}
}

void draw_feature(vector<weak_cla>& ada, vector<vector<int> >& ftr)
{
	int h_n[5] = {8704, 17408, 22848, 28288, 32384}; // 2h, 2v, 3h, 3v, 4
	for(int k = 0; k < ada.size(); k++)
	{
		Mat img = imread("../data/newface16/face16_000001.bmp");
		int ind = ada[k].index;
		double i = ftr[ind][0], j = ftr[ind][1], p = ftr[ind][2], q = ftr[ind][3];
		if(ind < h_n[0])
		{
			draw_sq(i,j,p,(p+q)/2, img, 1);
			draw_sq(i,j,(p+q)/2,q, img, 0);
		}
		else if(ind < h_n[1])
		{
			draw_sq(i,(i+j)/2,p,q, img,1);
			draw_sq((i+j)/2,j,p,q, img,0);
		}
		else if(ind < h_n[2])
		{
			draw_sq(i,j,p,p*2/3+q/3, img,1);
			draw_sq(i,j,p/3+q*2/3,q, img,1);
			draw_sq(i,j,p*2/3+q/3,p/3+q*2/3, img,0);
		}
		else if(ind < h_n[3])
		{
			draw_sq(i,i*2/3+j/3,p,q,img,1);
			draw_sq(i/3+j*2/3,j,p,q,img,1);
			draw_sq(i*2/3+j/3,i/3+j*2/3,p,q,img,0);
		}
		else
		{
			draw_sq(i,(i+j)/2,p,(p+q)/2,img,1);
			draw_sq((i+j)/2,j,(p+q)/2,q, img,1);
			draw_sq(i,(i+j)/2,(p+q)/2,q,img,0);
			draw_sq((i+j)/2,j,p,(p+q)/2,img,0);
		}

		imwrite(to_string(k) + ".jpg", img);
	}
}
