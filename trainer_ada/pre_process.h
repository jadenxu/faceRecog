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

void read_img(vector<MatrixXd>& i_faces, vector<MatrixXd>& i_nfaces)
{
	vector<Mat> faces;
	faces.resize(NUM_F);
	vector<MatrixXd> s_faces;

	// read the faces and calculate the integral image
	for(int i = 1; i <= NUM_F; i++)
	{
		string file_name = to_string(i);
		int tem = file_name.length();
		for(int j = 0; j < 6-tem; j++)
			file_name = "0" + file_name;
		file_name = "../data/newface16/face16_" + file_name + ".bmp";
		faces[i-1] = imread(file_name);
	}

	for(int k = 0; k < NUM_F; k++)
	{
		MatrixXd tem_m = MatrixXd::Zero(17,17);
		for(int i = 1; i <= 16; i++)
		{
			for(int j = 1; j <= 16; j++)
			{
				tem_m(i,j) = tem_m(i,j-1) + faces[k].at<Vec3b>(i-1,j-1)[0];
			}
		}
		s_faces.push_back(tem_m);
	}
	
	for(int k = 0; k < NUM_F; k++)
	{
		MatrixXd tem_m = MatrixXd::Zero(17,17);
		for(int i = 1; i <= 16; i++)
		{
			for(int j = 1; j <= 16; j++)
			{
				tem_m(i,j) = tem_m(i-1,j) + s_faces[k](i,j);
			}
		}
		i_faces.push_back(tem_m);
	}
	faces.clear();
	s_faces.clear();

	vector<Mat> nfaces;
	nfaces.resize(NUM_NF);
	vector<MatrixXd> s_nfaces;
	
	// read the faces and calculate the integral image
	for(int i = 1; i <= NUM_NF; i++)
	{
		string file_name = to_string(i);
		int tem = file_name.length();
		for(int j = 0; j < 6-tem; j++)
			file_name = "0" + file_name;
		file_name = "../data/nonface16/nonface16_" + file_name + ".bmp";
		nfaces[i-1] = imread(file_name);
	}
	
	for(int k = 0; k < NUM_NF; k++)
	{
		MatrixXd tem_m = MatrixXd::Zero(17,17);
		for(int i = 1; i <= 16; i++)
		{
			for(int j = 1; j <= 16; j++)
			{
				tem_m(i,j) = tem_m(i,j-1) + nfaces[k].at<Vec3b>(i-1,j-1)[0];
			}
		}
		s_nfaces.push_back(tem_m);
	}
	
	for(int k = 0; k < NUM_NF; k++)
	{
		MatrixXd tem_m = MatrixXd::Zero(17,17);
		for(int i = 1; i <= 16; i++)
		{
			for(int j = 1; j <= 16; j++)
			{
				tem_m(i,j) = tem_m(i-1,j) + s_nfaces[k](i,j);
			}
		}
		i_nfaces.push_back(tem_m);
	}
	s_nfaces.clear();
	nfaces.clear();
}

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

void feature_value(
  vector<MatrixXd>& i_faces, 
  vector<MatrixXd>& i_nfaces, 
  vector<vector<int> >& ftr, 
  vector<vector<int> >& f_val, 
  vector<vector<int> >& nf_val)
{
	int h_n[5] = {8704, 17408, 22848, 28288, 32384}; // 2h, 2v, 3h, 3v, 4
	int i,  j, p, q;

	for(int n = 0; n < h_n[0]; n++)
	{
		for(int m = 0; m < NUM_F; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			f_val[n][m] = cal_sq(i,j,p,(p+q)/2, i_faces[m]) - cal_sq(i,j,(p+q)/2,q, i_faces[m]);
		}
		for(int m = 0; m < NUM_NF; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			nf_val[n][m] = cal_sq(i,j,p,(p+q)/2, i_nfaces[m]) - cal_sq(i,j,(p+q)/2,q, i_nfaces[m]);
		}
	}

	for(int n = h_n[0]; n < h_n[1]; n++)
	{
		for(int m = 0; m < NUM_F; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			f_val[n][m] = cal_sq(i,(i+j)/2,p,q, i_faces[m]) - cal_sq((i+j)/2,j,p,q, i_faces[m]);
		}
		for(int m = 0; m < NUM_NF; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			nf_val[n][m] = cal_sq(i,(i+j)/2,p,q, i_nfaces[m]) - cal_sq((i+j)/2,j,p,q, i_nfaces[m]);
		}
	}
	
	for(int n = h_n[1]; n < h_n[2]; n++)
	{
		for(int m = 0; m < NUM_F; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			f_val[n][m] = cal_sq(i,j,p,(p*2+q)/3, i_faces[m]) + cal_sq(i,j,(p+q*2)/3,q, i_faces[m]) - cal_sq(i,j,(p*2+q)/3,(p+q*2)/3, i_faces[m]);
		}
		for(int m = 0; m < NUM_NF; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			nf_val[n][m] = cal_sq(i,j,p,(p*2+q)/3, i_nfaces[m]) + cal_sq(i,j,(p+q*2)/3,q, i_nfaces[m]) - cal_sq(i,j,(p*2+q)/3,(p+q*2)/3, i_nfaces[m]);
		}
	}

	for(int n = h_n[2]; n < h_n[3]; n++)
	{
		for(int m = 0; m < NUM_F; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			f_val[n][m] = cal_sq(i,(i*2+j)/3,p,q,i_faces[m]) + cal_sq((i+j*2)/3,j,p,q,i_faces[m]) - cal_sq((i*2+j)/3,(i+j*2)/3,p,q,i_faces[m]);
		}
		for(int m = 0; m < NUM_NF; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			nf_val[n][m] = cal_sq(i,(i*2+j)/3,p,q,i_nfaces[m]) + cal_sq((i+j*2)/3,j,p,q,i_nfaces[m]) - cal_sq((i*2+j)/3,(i+j*2)/3,p,q,i_nfaces[m]);
		}
	}

	for(int n = h_n[3]; n < h_n[4]; n++)
	{
		for(int m = 0; m < NUM_F; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			f_val[n][m] = cal_sq(i,(i+j)/2,p,(p+q)/2,i_faces[m]) + cal_sq((i+j)/2,j,(p+q)/2,q, i_faces[m]) - cal_sq(i,(i+j)/2,(p+q)/2,q,i_faces[m]) - cal_sq((i+j)/2,j,p,(p+q)/2,i_faces[m]);
		}
		for(int m = 0; m < NUM_NF; m++)
		{
			i = ftr[n][0], j = ftr[n][1], p = ftr[n][2], q = ftr[n][3];
			nf_val[n][m] = cal_sq(i,(i+j)/2,p,(p+q)/2,i_nfaces[m]) + cal_sq((i+j)/2,j,(p+q)/2,q, i_nfaces[m]) - cal_sq(i,(i+j)/2,(p+q)/2,q,i_nfaces[m]) - cal_sq((i+j)/2,j,p,(p+q)/2,i_nfaces[m]);
		}
	}
}
