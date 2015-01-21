#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<ctime>
#include "pre_process.h"

int main()
{	
	int t1= time(0);
	vector<vector<int> > ftr;
	ftr.resize(32384);
	for(int i = 0; i < 32384; i++)
		ftr[i].resize(4);
	find_feature(ftr);

	int T = 100;
	vector<weak_cla> ada; 
	ada.resize(T);
	ifstream input;
	input.open("ada.txt");
	for(int i = 0; i < 100; i++)
	{
		input>>ada[i].index>>ada[i].pol>>ada[i].alpha>>ada[i].theta;
	}
	input.close();

	Mat img = imread("../data/class_photo_2013.jpg");
	Mat gimg;
	cvtColor(img, gimg, CV_BGR2GRAY);
	vector<point> v;

	//bool ok = false;
	ofstream output;
	int size = 16;
	output.open("tem.txt");
	int b[5] = {64,50,32,16,20};
	double c[5] = {5, 3.5, 5, 5, 5};
	double d[5] = {11, 4.5, 11,11,11};
	for(int a = 4; a >= 0; a--)
	{
		size = b[a];
		cout<<size<<" "<<v.size()<<endl;
		for(int i = 200; i < gimg.rows - size/2; i+=size/4)
		{
			for(int j = 100; j < gimg.cols - size/2; j+=size/4)
			{
				Mat window(size, size, CV_8UC1);
				// get the window, min of window and max of window
				int min = INT_MAX, max = INT_MIN;
				for(int p = 0; p < size; p++)
				{
					for(int q = 0; q < size; q++)
					{
						window.at<uchar>(p,q) = gimg.at<uchar>(i-size/2+p,j-size/2+q);
						if(window.at<uchar>(p,q) < min)
							min = window.at<uchar>(p,q);
						if(window.at<uchar>(p,q) > max)
							max = window.at<uchar>(p,q);
					}
				}
				if(max - min < 80)
					continue;

				resize(window, window, Size(16,16), 0, 0, INTER_LINEAR);
				equalizeHist(window, window);

				//calculate the F(x)
				double tem_sum = 0;
				for(int k = 0; k < T; k++)
				{
					int f_value = feature_value(ada[k].index, window, ftr);
					tem_sum += ada[k].pol * ada[k].alpha * double(f_value - ada[k].theta >= 0 ? 1 : -1);
					//cout<<tem_sum<<endl;
				}
				if(tem_sum > 0)
					output<<i<<" "<<j<<" "<<tem_sum<<endl;
				if(tem_sum > c[a] && tem_sum < d[a])
				{
					cout<<i<<" "<<j<<" "<<v.size()<<" "<<tem_sum<<endl;
					bool ok = true;
					for(int k = 0; k < v.size(); k++)
					{
						if(i-size/2 >= v[k].x - v[k].num/2 && i+size/2 < v[k].x + v[k].num/2 && j-size/2 >= v[k].y - v[k].num/2 && j+size/2 < v[k].y + v[k].num/2)
						{
							ok = false;
							break;
						}
					}
					if(ok)
					{
						v.push_back(point(i,j,size));
					}
				}
			}
		}
	}
	output.close();
	
	for(int i = 0; i < v.size(); i++)
	{
		int x, y, n;
		x = v[i].x; y = v[i].y; n = v[i].num;
		int lx = x-n/2, ly = y-n/2;
		if(n % 2 == 0)
		{
			lx--;
			ly--;
		}
		rectangle(img, Point(ly,lx), Point(ly+n, lx+n), Scalar(0,255,255), 2, CV_AA);
	}
	imwrite("result.jpg", img);
	int t2 = time(0);
	cout<<"time "<<t2-t1<<endl;
	//waitKey();
	
  return 0;
}
