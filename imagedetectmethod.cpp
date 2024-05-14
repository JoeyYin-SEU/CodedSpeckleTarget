#include "imagedetectmethod.h"
#include <imgproc/types_c.h>

Image2D_InterpCoef::Image2D_InterpCoef()
{
	eg_mat = nullptr;
}
Image2D_InterpCoef::Image2D_InterpCoef(int width, int height)
{
	eg_mat = nullptr;
	InterpCoef_struct* ptr1d = (InterpCoef_struct*)calloc(width * height
		, sizeof(InterpCoef_struct));
	eg_mat = (InterpCoef_struct**)malloc(height * sizeof(InterpCoef_struct*));

	for (int i = 0; i < height; i++)
	{
		eg_mat[i] = ptr1d + i * width;
	}

	this->width = width;
	this->height = height;
}
void Image2D_InterpCoef::Delete2D_InterpCoef()
{
	if (eg_mat == nullptr)
		return;
	free(eg_mat[0]);
	free(eg_mat);
	eg_mat = nullptr;
}
Image2D_InterpCoef::~Image2D_InterpCoef()
{
}

ImageDetectMethod::ImageDetectMethod(QObject* parent)
	: QObject(parent)
{
}

ImageDetectMethod::~ImageDetectMethod()
{
}



inline double ImageDetectMethod::getAmplitude(cv::Mat& dx, cv::Mat& dy, int i, int j)
{
	cv::Point2d mag(dx.at<float>(i, j), dy.at<float>(i, j));
	return norm(mag);
}

inline void ImageDetectMethod::getMagNeighbourhood(cv::Mat& dx, cv::Mat& dy, cv::Point& p, int w, int h, std::vector<double>& mag)
{
	int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
	int down = p.y + 1 < h ? p.y + 1 : p.y;
	int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
	int right = p.x + 1 < w ? p.x + 1 : p.x;

	mag[0] = getAmplitude(dx, dy, top, left);
	mag[1] = getAmplitude(dx, dy, top, p.x);
	mag[2] = getAmplitude(dx, dy, top, right);
	mag[3] = getAmplitude(dx, dy, p.y, left);
	mag[4] = getAmplitude(dx, dy, p.y, p.x);
	mag[5] = getAmplitude(dx, dy, p.y, right);
	mag[6] = getAmplitude(dx, dy, down, left);
	mag[7] = getAmplitude(dx, dy, down, p.x);
	mag[8] = getAmplitude(dx, dy, down, right);
}

inline void ImageDetectMethod::get2ndFacetModelIn3x3(std::vector<double>& mag, std::vector<double>& a)
{
	a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) / 9.0;
	a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
	a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
	a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] + mag[6] - 2.0 * mag[7] + mag[8]) / 6.0;
	a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0;
	a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) + mag[6] + mag[7] + mag[8]) / 6.0;
}
/*
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values.
*/
inline void ImageDetectMethod::eigenvals(std::vector<double>& a, double eigval[2], double eigvec[2][2])
{
	// derivatives
	// fx = a[1], fy = a[2]
	// fxy = a[4]
	// fxx = 2 * a[3]
	// fyy = 2 * a[5]
	double dfdrc = a[4];
	double dfdcc = a[3] * 2.0;
	double dfdrr = a[5] * 2.0;
	double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

	/* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
	if (dfdrc != 0.0) {
		theta = 0.5 * (dfdcc - dfdrr) / dfdrc;
		t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
		if (theta < 0.0) t = -t;
		c = 1.0 / sqrt(t * t + 1.0);
		s = t * c;
		e1 = dfdrr - t * dfdrc;
		e2 = dfdcc + t * dfdrc;
	}
	else {
		c = 1.0;
		s = 0.0;
		e1 = dfdrr;
		e2 = dfdcc;
	}
	n1 = c;
	n2 = -s;

	/* If the absolute value of an eigenvalue is larger than the other, put that
	eigenvalue into first position.  If both are of equal absolute value, put
	the negative one first. */
	if (fabs(e1) > fabs(e2)) {
		eigval[0] = e1;
		eigval[1] = e2;
		eigvec[0][0] = n1;
		eigvec[0][1] = n2;
		eigvec[1][0] = -n2;
		eigvec[1][1] = n1;
	}
	else if (fabs(e1) < fabs(e2)) {
		eigval[0] = e2;
		eigval[1] = e1;
		eigvec[0][0] = -n2;
		eigvec[0][1] = n1;
		eigvec[1][0] = n1;
		eigvec[1][1] = n2;
	}
	else {
		if (e1 < e2) {
			eigval[0] = e1;
			eigval[1] = e2;
			eigvec[0][0] = n1;
			eigvec[0][1] = n2;
			eigvec[1][0] = -n2;
			eigvec[1][1] = n1;
		}
		else {
			eigval[0] = e2;
			eigval[1] = e1;
			eigvec[0][0] = -n2;
			eigvec[0][1] = n1;
			eigvec[1][0] = n1;
			eigvec[1][1] = n2;
		}
	}
}

cv::Mat ImageDetectMethod::Image_Resize(const cv::Mat& ori_image_mat, int hei, int wid, int method, bool anti)
{
	coder::array<double, 2U> binary_I;
	binary_I.set_size(hei, wid);
	coder::array<double, 2U> b_I;
	b_I.set_size(ori_image_mat.rows, ori_image_mat.cols);
	for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
		for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
			b_I[idx0 + b_I.size(0) * idx1] = ori_image_mat.at<double>(idx0, idx1);
		}
	}
	get_resize(b_I, hei, wid, method, anti, binary_I);
	cv::Mat Res = cv::Mat(hei, wid, CV_32FC1);
	for (int idx0{ 0 }; idx0 < binary_I.size(0); idx0++) {
		for (int idx1{ 0 }; idx1 < binary_I.size(1); idx1++)
		{
			Res.at<float>(idx0, idx1) = binary_I[idx0 + binary_I.size(0) * idx1];
		}
	}
	return Res;
}
inline double ImageDetectMethod::vector2angle(double x, double y)
{
	double a = std::atan2(y, x);
	return a >= 0.0 ? a : a + CV_2PI;
}
int greater(double a, double b)
{
	if (a <= b) return FALSE;  /* trivial case, return as soon as possible */
	if ((a - b) < 1000 * DBL_EPSILON) return FALSE;
	return TRUE; /* greater */
}
void ImageDetectMethod::extractSubPixPoints(cv::Mat& dx, cv::Mat& dy, std::vector<std::vector<cv::Point> >& contoursInPixel
	, std::vector<std::vector<cv::Point2f>>& contours)
{
	contours.clear();
	int w = dx.cols;
	int h = dx.rows;
	for (size_t i = 0; i < contoursInPixel.size(); ++i)
	{
		std::vector<cv::Point2f> con_temp;
		for (size_t j = 0; j < contoursInPixel[i].size(); ++j)
		{
			if (contoursInPixel[i][j].x<1 || contoursInPixel[i][j].x >=(w - 1)
				|| contoursInPixel[i][j].y<1 || contoursInPixel[i][j].y >=(h - 1))
			{

				continue;
			}
			int Dx = 0;                     /* interpolation is along Dx,Dy		*/
			int Dy = 0;                     /* which will be selected below		*/
			double mod = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x), 2) 
				+ pow(dy.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x), 2));
			double L = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x - 1), 2)
				+ pow(dy.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x - 1), 2));   /* modG at pixel on the left			*/
			double R = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x + 1), 2)
				+ pow(dy.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x + 1), 2));  /* modG at pixel on the right		*/
			double U = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y + 1, contoursInPixel[i][j].x), 2)
				+ pow(dy.at<float>(contoursInPixel[i][j].y + 1, contoursInPixel[i][j].x), 2)); /* modG at pixel up					*/
			double D = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y - 1, contoursInPixel[i][j].x), 2)
				+ pow(dy.at<float>(contoursInPixel[i][j].y - 1, contoursInPixel[i][j].x), 2)); ; /* modG at pixel below				*/
			double gx = fabs(dx.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x));  /* absolute value of Gx				*/
			double gy = fabs(dy.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x));  /* absolute value of Gy				*/
			if (greater(mod, L) && !greater(R, mod) && gx >= gy)
			{
				Dx = 1; /* H */
			}
			else if (greater(mod, D) && !greater(U, mod) && gx <= gy)
			{
				Dy = 1;
			}

			if (Dx > 0 || Dy > 0)
			{
				/* offset value is in [-0.5, 0.5] */
				double a = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y - Dy, contoursInPixel[i][j].x - Dx), 2)
					+ pow(dy.at<float>(contoursInPixel[i][j].y - Dy, contoursInPixel[i][j].x - Dx), 2));
				double b = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x), 2)
					+ pow(dy.at<float>(contoursInPixel[i][j].y, contoursInPixel[i][j].x), 2));
				double c = sqrt(pow(dx.at<float>(contoursInPixel[i][j].y + Dy, contoursInPixel[i][j].x + Dx), 2)
					+ pow(dy.at<float>(contoursInPixel[i][j].y + Dy, contoursInPixel[i][j].x + Dx), 2));
				double offset = 0.5 * (a - c) / (a - b - b + c);
				con_temp.push_back(cv::Point2f(contoursInPixel[i][j].x + offset * Dx, contoursInPixel[i][j].y + offset * Dy));
			}
		}
		contours.push_back(con_temp);
	}
}



bool ImageDetectMethod::ImagePreprocess(const cv::Mat& ori_image_mat, cv::Mat& processed_image_mat)
{
	GaussianBlur(ori_image_mat, processed_image_mat, cv::Size(5, 5), 1, 1);
	return true;
}
void ImageDetectMethod::Sub_pixel_edge(const cv::Mat image_mat, std::vector<std::vector<cv::Point2f>>& Cons,double sigma
	, DetectContoursMethod type)
{
	cv::Mat threshold_image;
	std::vector<std::vector<cv::Point>> contours_old;
	std::vector<std::vector<cv::Point>> contours;
	threshold_image = image_mat.clone();
	if (threshold_image.channels() == 3)
	{
		cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
	}
	if (type == Sobel_Method)
	{
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 0, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
	}
	else if (type == CANNY_Method)
	{
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 1, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
	}
	else if (type == Prewitt_Method)
	{
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 2, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
	}
	else if (type == Roberts_Method)
	{
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 3, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
	}
	else if (type == OTSU_Method)
	{
		threshold(image_mat, threshold_image, 0, 255.0, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
	}
	else if (type == ADAPTIVE_THRESH_Method)
	{
		adaptiveThreshold(image_mat, threshold_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0);
	}
	else if (type == Block_OTSU)
	{
		threshold(image_mat, threshold_image, 0, 255.0, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
	}
	std::vector<cv::Vec4i> hierarchy;
	findContours(threshold_image, contours_old, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));



	for (int ii = 0; ii < hierarchy.size(); ii++)
	{
		if (hierarchy[ii][2] == -1)
		{
			contours.push_back(contours_old[ii]);
		}
	}
	threshold_image = image_mat.clone();
	if (threshold_image.channels() == 3)
	{
		cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
	}
	coder::array<float, 2U> dx,dy;
	dx.set_size(threshold_image.rows, threshold_image.cols);
	dy.set_size(threshold_image.rows, threshold_image.cols);
	coder::array<boolean_T, 2U> b_I;
	b_I.set_size(threshold_image.rows, threshold_image.cols);
	for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
		for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
			b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
		}
	}
	get_fx_fy(b_I, sigma, dx, dy);
	cv::Mat dx_image(threshold_image.rows, threshold_image.cols, CV_32FC1);
	cv::Mat dy_image(threshold_image.rows, threshold_image.cols, CV_32FC1);
	for (int idx0{ 0 }; idx0 < dx.size(0); idx0++) {
		for (int idx1{ 0 }; idx1 < dx.size(1); idx1++)
		{
			dx_image.at<float>(idx0, idx1) = dx[idx0 + b_I.size(0) * idx1];
			dy_image.at<float>(idx0, idx1) = dy[idx0 + b_I.size(0) * idx1];
		}
	}
	extractSubPixPoints(dx_image, dy_image, contours, Cons);
}
bool ImageDetectMethod::DetectClosedContours(const cv::Mat& ori_image_mat, std::vector<std::vector<cv::Point>>& contours
	, DetectContoursMethod image_process_method)
{
	//vector<vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat threshold_image;
	if (image_process_method == Sobel_Method)
	{
		threshold_image = ori_image_mat.clone();
		if (threshold_image.channels() == 3)
		{
			cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
		}
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 0, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == CANNY_Method)
	{
		threshold_image = ori_image_mat.clone();
		if (threshold_image.channels() == 3)
		{
			cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
		}
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 1, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) 
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == Prewitt_Method)
	{
		threshold_image = ori_image_mat.clone();
		if (threshold_image.channels() == 3)
		{
			cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
		}
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 2, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}	
		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == Roberts_Method)
	{
		threshold_image = ori_image_mat.clone();
		if (threshold_image.channels() == 3)
		{
			cv::cvtColor(threshold_image, threshold_image, CV_BGR2GRAY);
		}
		coder::array<unsigned char, 2U> binary_I;
		binary_I.set_size(threshold_image.rows, threshold_image.cols);
		coder::array<boolean_T, 2U> b_I;
		b_I.set_size(threshold_image.rows, threshold_image.cols);
		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
				b_I[idx0 + b_I.size(0) * idx1] = threshold_image.at<uchar>(idx0, idx1);
			}
		}
		imedge_2d(b_I, 3, binary_I);

		for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
			for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++)
			{
				if (binary_I[idx0 + b_I.size(0) * idx1])
				{
					threshold_image.at<uchar>(idx0, idx1) = 255;
				}
				else
				{
					threshold_image.at<uchar>(idx0, idx1) = 0;
				}
			}
		}
		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == OTSU_Method)
	{
		threshold(ori_image_mat, threshold_image, 0, 255.0, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == ADAPTIVE_THRESH_Method)
	{
		adaptiveThreshold(ori_image_mat, threshold_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0);
		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	else if (image_process_method == Block_OTSU)
	{
		threshold(ori_image_mat, threshold_image, 0, 255.0, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

		findContours(threshold_image, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	}
	//cv::namedWindow("canny", cv::WINDOW_NORMAL);
	//cv::imshow("canny", threshold_image);
	//cv::waitKey(0);
	return true;
}

float ImageDetectMethod::Contours_arc(std::vector<cv::Point> C , cv::Point2f center)
{
	if (C.size() < 2)
	{
		return 0;
	}
	std::vector<float> arc_vec;
	for (int ii = 0; ii < C.size(); ii++)
	{
		arc_vec.push_back(atan2f((float)C[ii].x - center.x, (float)C[ii].y - center.y));
	}
	std::sort(arc_vec.begin(), arc_vec.end());
	double arc_max = arc_vec[0] + 6.283185307179586 - arc_vec[arc_vec.size() - 1];
	for (int ii = 0; ii < C.size() - 1; ii++)
	{
		double L_now = arc_vec[ii + 1] - arc_vec[ii];
		arc_max = arc_max < L_now ? L_now : arc_max;
	}
	return (6.283185307179586 - arc_max) / 6.283185307179586 * 360.0;
}
float ImageDetectMethod::Contours_Length(std::vector<cv::Point> C)
{
	if (C.size() < 2)
	{
		return 0;
	}
	double L_max = sqrt(pow(C[0].x - C[C.size() - 1].x, 2) + pow(C[0].y - C[C.size() - 1].y, 2));
	double L_sum = L_max;
	for (int ii = 0; ii < C.size() - 1; ii++)
	{
		double L_now = sqrt(pow(C[ii].x - C[ii + 1].x, 2) + pow(C[ii].y - C[ii + 1].y, 2));
		L_max = L_max < L_now ? L_now : L_max;
		L_sum += L_now;
	}
	return L_sum - L_max;
}
bool ImageDetectMethod::FilterEllipseContours(const std::vector<std::vector<cv::Point>>& contours
	, int min_radius_pixel, int max_radius_pixel,
	float ellipse_error_pixel, 
	QList<QList<float>>& ellipse_pars, double max_aspect_ratio, int min_points,int min_contour_num,
	int min_arc)
{
	//ellipse_error_pixel = 0.5;

	cv::Mat ori_image = 255 * cv::Mat::zeros(3648, 5472, CV_8U);
	min_points = min_points < 6 ? 6 : min_points;
	ellipse_pars.clear();
	for (int i = 0; i < contours.size(); i++)
	{
		//椭圆最小6点拟合
		int count = contours[i].size();
		if (count < min_points)
		{
			continue;
		}

		//周长,面积
		double length = arcLength(contours[i], true);
		double area = contourArea(contours[i]);
		double perimeter = cv::arcLength(contours[i], true);
		std::vector<cv::Point> contours_ploy;
		cv::approxPolyDP(contours[i], contours_ploy, 0.02 * perimeter, true);
		int CornerNum = contours_ploy.size();
		if (CornerNum < min_contour_num)
		{
			continue;
		}
		cv::RotatedRect rRect = fitEllipseAMS(contours[i]);
		if (isnan(rRect.center.x) || isinf(rRect.center.x))
		{
			continue;
		}
		if (isnan(rRect.center.y) || isinf(rRect.center.y))
		{
			continue;
		}
		if (isnan(rRect.size.width) || isinf(rRect.size.width))
		{
			continue;
		}
		if (isnan(rRect.size.height) || isinf(rRect.size.height))
		{
			continue;
		}

		//半径大小判断
		if (max_radius_pixel < rRect.size.width /2.0 || min_radius_pixel > rRect.size.height / 2.0)
		{
			continue;
		}

		////////////////长短轴比例判断
		if (rRect.size.height / rRect.size.width > max_aspect_ratio)
		{
			continue;
		}

		////////////////最小二乘拟合误差判断，半径差值


		float whole_error = 0;
		for (int j = 0; j < count; j++)
		{
			whole_error += ErrorDROfEllipseFit(rRect.center.x, rRect.center.y, rRect.size.width * 0.5, rRect.size.height * 0.5, rRect.angle * M_PI / 180,
				float(contours[i][j].x), float(contours[i][j].y));
		}
		float aver_error = whole_error / count;

		if (aver_error > ellipse_error_pixel * 1.5)
		{
			continue;
		}
		if (Contours_arc(contours[i], rRect.center)< min_arc)
		{
			continue;
		}
		////二次优化
		//double* coff_now = new double[5];
		//coff_now[0] = rRect.center.x;
		//coff_now[1] = rRect.center.y;
		//coff_now[2] = rRect.size.width / 2.0;
		//coff_now[3] = rRect.size.height / 2.0;
		//coff_now[4] = rRect.angle * M_PI / 180;
		//ceres::Problem problem;
		//for (int jj = 0; jj < contours[i].size(); jj++)
		//{
		//	Eigen::Vector2d ellipse_point(contours[i][jj].x, contours[i][jj].y);
		//	ceres::CostFunction* cost_function = Fit_ellipse_ceres::Create(ellipse_point);
		//	ceres::LossFunction* loss_function = new ceres::CauchyLoss(1);
		//	problem.AddResidualBlock(cost_function, loss_function, coff_now);
		//}
		//ceres::Solver::Summary summary;
		//ceres::Solver::Options options;
		//options.minimizer_progress_to_stdout = false;
		//options.max_num_iterations = 1000;
		//options.function_tolerance = 1e-9;
		//options.gradient_tolerance = 1e-9;
		//options.parameter_tolerance = 1e-9;
		//ceres::Solve(options, &problem, &summary);


		//cv::RotatedRect rRect_ceres;
		//rRect_ceres.center.x = coff_now[0];
		//rRect_ceres.center.y = coff_now[1];
		//rRect_ceres.angle = coff_now[4] / M_PI * 180;
		//rRect_ceres.size.width = 2 * coff_now[2];
		//rRect_ceres.size.height = 2 * coff_now[3];
		//float whole_error_ceres = 0;
		//for (int j = 0; j < count; j++)
		//{
		//	whole_error_ceres += ErrorDROfEllipseFit(rRect_ceres.center.x, rRect_ceres.center.y, rRect_ceres.size.width * 0.5, rRect_ceres.size.height * 0.5, rRect_ceres.angle * M_PI / 180,
		//		float(contours[i][j].x), float(contours[i][j].y));
		//}
		//float aver_error_ceres = whole_error_ceres / count;

		////if (aver_error_ceres > ellipse_error_pixel)
		////{
		////	continue;
		////}
		//QList<float> ellipse_par_list;
		//ellipse_par_list << rRect_ceres.center.x;
		//ellipse_par_list << rRect_ceres.center.y;
		//ellipse_par_list << rRect_ceres.size.width * 0.5;
		//ellipse_par_list << rRect_ceres.size.height * 0.5;
		//ellipse_par_list << rRect_ceres.angle;
		//ellipse_par_list << aver_error_ceres;
		//ellipse_par_list << i;

		QList<float> ellipse_par_list;
		ellipse_par_list << rRect.center.x;
		ellipse_par_list << rRect.center.y;
		ellipse_par_list << rRect.size.width * 0.5;
		ellipse_par_list << rRect.size.height * 0.5;
		ellipse_par_list << rRect.angle * M_PI / 180;
		ellipse_par_list << aver_error;
		ellipse_par_list << i;
		ellipse_pars.append(ellipse_par_list);
	}
	return true;
}

float ImageDetectMethod::ErrorDROfEllipseFit(float center_x, float center_y, float ellipse_a, float ellipse_b,
	float ellipse_angle_in_pi, float x, float y)
{
	//坐标转换，图像坐标系转椭圆坐标系      x^2/a^2 + y^2/b^2 =1
	float tr_x = (x - center_x) * cos(ellipse_angle_in_pi) + (y - center_y) * sin(ellipse_angle_in_pi);
	float tr_y = -(x - center_x) * sin(ellipse_angle_in_pi) + (y - center_y) * cos(ellipse_angle_in_pi);

	//计算拟合误差，半径差作为拟合误差
	float alfa = atan2(tr_y, tr_x);

	float r = ellipse_a * ellipse_b / sqrt(ellipse_a * ellipse_a * sin(alfa) * sin(alfa) + ellipse_b * ellipse_b * cos(alfa) * cos(alfa));
	float delta_r = sqrt(tr_x * tr_x + tr_y * tr_y) - r;

	return abs(delta_r);
}

float ImageDetectMethod::LeastSquareErrorOfEllipseFit(float center_x, float center_y, float ellipse_a, float ellipse_b,
	float ellipse_angle_in_pi, float x, float y)
{
	//坐标转换，图像坐标系转椭圆坐标系          x^2/a^2 + y^2/b^2 =1
	float tr_x = (x - center_x) * cos(ellipse_angle_in_pi) + (y - center_y) * sin(ellipse_angle_in_pi);
	float tr_y = -(x - center_x) * sin(ellipse_angle_in_pi) + (y - center_y) * cos(ellipse_angle_in_pi);

	//计算拟合误差，最小二乘拟合误差
	float e = tr_x * tr_x / ellipse_a / ellipse_a + tr_y * tr_y / ellipse_b / ellipse_b - 1;

	return e * e;
}

bool ImageDetectMethod::FilterEllipseContoursForCodePoint(const cv::Mat& image_mat, float ratio_k, float ratio_k1, float ratio_k2,
	QList<QList<float>>& ellipse_pars,
	float delta_Mt, float fore_stdDev, float back_stdDev)//code uncode的区分，C
{
	//进一步筛选出用编码点解码的点
	//灰度准则
	for (int i = 0; i < ellipse_pars.size(); i++)
	{
		float out_foreground_mean = 0;
		float out_background_mean = 0;
		float out_foreground_std = 0;
		float out_background_std = 0;
		bool is_gray_judge = EllipseGrayJudgeForCodePoint(image_mat, ellipse_pars[i][0], ellipse_pars[i][1],
			ellipse_pars[i][2], ellipse_pars[i][3], ellipse_pars[i][4], ratio_k, out_foreground_mean, out_background_mean
			, out_foreground_std, out_background_std, delta_Mt, fore_stdDev, back_stdDev);

		if (is_gray_judge == false)
		{
			ellipse_pars.removeAt(i);
			i--;
		}
	}
	//剔除小的编码带影响
	///////筛选出标志点进行解码,位置关系，剔除外圆环和小编码带
	for (int i = 0; i < ellipse_pars.size() - 1; i++)
	{
		for (int j = i + 1; j < ellipse_pars.size(); j++)
		{
			float x1 = ellipse_pars[i][0];
			float y1 = ellipse_pars[i][1];
			float x2 = ellipse_pars[j][0];
			float y2 = ellipse_pars[j][1];
			float length_of_2_points = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

			if (length_of_2_points < std::min(ellipse_pars[i][3], ellipse_pars[j][3]))
			{
				if (ellipse_pars[i][3] > ellipse_pars[j][3])
				{
					ellipse_pars.removeAt(i);
					i--;
					break;
				}
				else
				{
					ellipse_pars.removeAt(j);
					j--;
				}
			}
			else
				if (length_of_2_points < std::min(ellipse_pars[i][3], ellipse_pars[j][3]) * ratio_k2)
				{
					if (ellipse_pars[i][5] / sqrt(ellipse_pars[i][2] * ellipse_pars[i][3]) > ellipse_pars[j][5] / sqrt(ellipse_pars[j][2] * ellipse_pars[j][3]))
					{
						ellipse_pars.removeAt(i);
						i--;
						break;
					}
					else
					{
						ellipse_pars.removeAt(j);
						j--;
					}
				}
		}
	}
	return true;
}
bool ImageDetectMethod::EllipseGrayJudgeForPointCSI_is2Circle(
	QList<QList<float>> ellipse_pars_all, QList<float> ellipse_pars_now, float rati_k)
{
	bool is_exist = false;
	for (int i = 0; i < ellipse_pars_all.size() - 1; i++)
	{
		if (ellipse_pars_all[i][6] == ellipse_pars_now[6])
		{
			continue;
		}
		float x1 = ellipse_pars_all[i][0];
		float y1 = ellipse_pars_all[i][1];
		float x2 = ellipse_pars_now[0];
		float y2 = ellipse_pars_now[1];
		float length_of_2_points = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

		if (length_of_2_points < std::min(ellipse_pars_all[i][3], ellipse_pars_now[3]))
		{
			double r1 = ellipse_pars_all[i][3] < ellipse_pars_now[3] ? ellipse_pars_all[i][3] : ellipse_pars_now[3];
			double r2 = ellipse_pars_all[i][3] > ellipse_pars_now[3] ? ellipse_pars_all[i][3] : ellipse_pars_now[3];
			double r_now = r1 / r2;
			if (r_now > rati_k * 0.5 && r_now < rati_k * 1.5)
			{
				is_exist = true;
				break;
			}
		}
	}
	return is_exist;
}
bool ImageDetectMethod::EllipseGrayJudgeForPointCSI(const cv::Mat& image_mat, float center_x, float center_y,
	float ellipse_a, float ellipse_b, float angle_in_pi, float ratio_k,
	float& out_std)
{
	float ellipse_a2;
	float ellipse_b2;

	if (ratio_k > 1)
	{
		ellipse_a2 = ellipse_a * ratio_k;
		ellipse_b2 = ellipse_b * ratio_k;
	}
	else
	{
		ellipse_a2 = ellipse_a;
		ellipse_b2 = ellipse_b;
		ellipse_a = ellipse_a2 * ratio_k;
		ellipse_b = ellipse_b2 * ratio_k;
	}

	int i_top = 0;
	int i_bottom = 0;
	int j_left = 0;
	int j_right = 0;
	if (int(center_y - ellipse_b2) < 0)
	{
		i_top = 0;
	}
	else i_top = int(center_y - ellipse_b2);
	if (int(center_y + ellipse_b2) > image_mat.rows - 1)
	{
		i_bottom = image_mat.rows - 1;
	}
	else i_bottom = int(center_y + ellipse_b2);
	if (int(center_x - ellipse_b2) < 0)
	{
		j_left = 0;
	}
	else j_left = int(center_x - ellipse_b2);
	if (int(center_x + ellipse_b2) > image_mat.cols - 1)
	{
		j_right = image_mat.cols - 1;
	}
	else j_right = int(center_x + ellipse_b2);

	float foreground_median = 0;
	float background_median = 0;

	QList<float> background_value_list;
	QList<float> foreground_value_list;
	QList<float> all_value_list;
	QList<float> whole_value_list;
	cv::Mat cropped_image;
	cv::Mat PROCE_img;
	for (int i = i_top; i <= i_bottom; i++)
	{
		for (int j = j_left; j <= j_right; j++)
		{
			const uchar* _image_mat_ptr = image_mat.ptr<uchar>(i);


			float tr_x = (j - center_x) * cos(angle_in_pi) + (i - center_y) * sin(angle_in_pi);
			float tr_y = -(j - center_x) * sin(angle_in_pi) + (i - center_y) * cos(angle_in_pi);

			if (tr_x * tr_x / int(ellipse_a2) / int(ellipse_a2) + tr_y * tr_y / int(ellipse_b2) / int(ellipse_b2) < 1 &&
				tr_x * tr_x / ellipse_a / ellipse_a + tr_y * tr_y / ellipse_b / ellipse_b > 1)
			{
				background_value_list.append(float(_image_mat_ptr[j]));
				//std::cout << 0 << "\t" << i << "\t" << j << "\t" << float(_image_mat_ptr[j]) << "\n";
			}
			else if (tr_x * tr_x / int(ellipse_a) / int(ellipse_a) + tr_y * tr_y / int(ellipse_b) / int(ellipse_b) < 1)
			{
				foreground_value_list.append(float(_image_mat_ptr[j]));
				//std::cout << 1 << "\t" << i << "\t" << j << "\t" << float(_image_mat_ptr[j]) << "\n";
			}
		}
	}
	int inter_1 = 1, inter_2 = 1;
	if (ratio_k * ratio_k - 1 > 1)
	{
		inter_1 = round(ratio_k * ratio_k - 1);
		inter_1 = inter_1 < 1 ? 1 : inter_1;
		inter_2 = 1;
	}
	else
	{
		inter_2 = round(1.0 / (ratio_k * ratio_k - 1));
		inter_2 = inter_2 < 1 ? 1 : inter_2;
		inter_1 = 1;
	}
	double mean_value = 0;
	double mean_value_number = 0;
	for (int ii = 0; ii < background_value_list.size(); ii = ii + inter_1)
	{
		all_value_list.push_back(background_value_list[ii]);
		mean_value += background_value_list[ii];
		mean_value_number += 1.0;
	}
	for (int ii = 0; ii < foreground_value_list.size(); ii = ii = ii + inter_2)
	{
		all_value_list.push_back(foreground_value_list[ii]);
		mean_value += foreground_value_list[ii];
		mean_value_number += 1.0;
	}
	mean_value /= mean_value_number;

	double std_value = 0;
	for (int ii = 0; ii < all_value_list.size(); ii++)
	{
		std_value += pow(all_value_list[ii] - mean_value, 2);
	}

	std_value /= mean_value_number;
	std_value = sqrt(std_value);
	out_std = std_value;

	return true;
}
bool ImageDetectMethod::EllipseGrayJudgeForCodePoint(const cv::Mat& image_mat, float center_x, float center_y,
	float ellipse_a, float ellipse_b, float angle_in_pi, float ratio_k,
	float& out_foreground_mean, float& out_background_mean, float& out_foreground_stdDev, float& out_background_stdDev,
	float delta_Mt, float fore_stdDev, float back_stdDev)
{
	float ellipse_a2 = ellipse_a * ratio_k;
	float ellipse_b2 = ellipse_b * ratio_k;

	int i_top = 0;
	int i_bottom = 0;
	int j_left = 0;
	int j_right = 0;
	if (int(center_y - ellipse_b2) < 0)
	{
		i_top = 0;
	}
	else i_top = int(center_y - ellipse_b2);
	if (int(center_y + ellipse_b2) > image_mat.rows - 1)
	{
		i_bottom = image_mat.rows - 1;
	}
	else i_bottom = int(center_y + ellipse_b2);
	if (int(center_x - ellipse_b2) < 0)
	{
		j_left = 0;
	}
	else j_left = int(center_x - ellipse_b2);
	if (int(center_x + ellipse_b2) > image_mat.cols - 1)
	{
		j_right = image_mat.cols - 1;
	}
	else j_right = int(center_x + ellipse_b2);

	float foreground_mean = 0;
	float background_mean = 0;
	float whole_mean = 0;
	float foreground_stdDev = 0;
	float background_stdDev = 0;
	float whole_stdDev = 0;
	int forground_num = 0;
	int background_num = 0;
	int whole_num = 0;

	QList<float> background_value_list;
	QList<float> foreground_value_list;
	QList<float> whole_value_list;
	cv::Mat cropped_image;
	cv::Mat PROCE_img;

	for (int i = i_top; i <= i_bottom; i++)
	{
		for (int j = j_left; j <= j_right; j++)
		{
			const uchar* _image_mat_ptr = image_mat.ptr<uchar>(i);
			//坐标转换图像坐标系转椭圆坐标系


			float tr_x = (j - center_x) * cos(angle_in_pi) + (i - center_y) * sin(angle_in_pi);
			float tr_y = -(j - center_x) * sin(angle_in_pi) + (i - center_y) * cos(angle_in_pi);
			//float tr_x = (j - center_x) * sin(angle_in_pi) - (i - center_y) * cos(angle_in_pi);
			//float tr_y = (j - center_x) * cos(angle_in_pi) + (i - center_y) * sin(angle_in_pi);

			if (tr_x * tr_x / int(ellipse_a2) / int(ellipse_a2) + tr_y * tr_y / int(ellipse_b2) / int(ellipse_b2) < 1 &&
				tr_x * tr_x / ellipse_a / ellipse_a + tr_y * tr_y / ellipse_b / ellipse_b > 1)
			{
				background_mean += float(_image_mat_ptr[j]);
				background_value_list.append(float(_image_mat_ptr[j]));
				background_num++;

				whole_mean += float(_image_mat_ptr[j]);
				whole_value_list.append(float(_image_mat_ptr[j]));
				whole_num++;
			}
			else if (tr_x * tr_x / int(ellipse_a) / int(ellipse_a) + tr_y * tr_y / int(ellipse_b) / int(ellipse_b) < 1)
			{
				foreground_mean += float(_image_mat_ptr[j]);
				foreground_value_list.append(float(_image_mat_ptr[j]));
				forground_num++;

				whole_mean += float(_image_mat_ptr[j]);
				whole_value_list.append(float(_image_mat_ptr[j]));
				whole_num++;
			}
		}
	}
	foreground_mean = foreground_mean / forground_num;
	background_mean = background_mean / background_num;
	whole_mean = whole_mean / whole_num;
	for (int i = 0; i < background_value_list.size(); i++)
	{
		background_stdDev += (background_value_list[i] - background_mean) * (background_value_list[i] - background_mean);
	}
	for (int i = 0; i < foreground_value_list.size(); i++)
	{
		foreground_stdDev += (foreground_value_list[i] - foreground_mean) * (foreground_value_list[i] - foreground_mean);
	}
	for (int i = 0; i < whole_value_list.size(); i++)
	{
		whole_stdDev += (whole_value_list[i] - whole_mean) * (whole_value_list[i] - whole_mean);
	}
	foreground_stdDev = sqrt(foreground_stdDev / forground_num);
	background_stdDev = sqrt(background_stdDev / background_num);
	whole_stdDev = sqrt(whole_stdDev / whole_num);
	out_foreground_mean = foreground_mean;
	out_background_mean = background_mean;
	out_foreground_stdDev = foreground_stdDev;
	out_background_stdDev = background_stdDev;
	if (abs(foreground_mean - background_mean) < delta_Mt)
	{
		return false;
	}
	if (foreground_stdDev > fore_stdDev)
	{
		return false;
	}
	if (background_stdDev > back_stdDev)
	{
		return false;
	}

	return true;
}

int* ImageDetectMethod::ReturnDefualtIdArray(int& array_size, CodePointBitesType code_bites_type /*=CodeBites15*/)
{
	switch (code_bites_type)
	{
	case CodeBites15:
		array_size = 429;
		//array_size = sizeof(static_15_8_code_id_false) / sizeof(static_15_8_code_id_false[0]);

		return &static_15_8_code_id_true[0];
		break;
	}

	return NULL;
}

bool ImageDetectMethod::Decoding20140210(const cv::Mat& image_mat, int& out_put_code_id, float center_x, float center_y,
	float ellipse_a, float ellipse_b, float angle_in_pi,
	float ratio_k1 /*=2.4*/, float ratio_k2 /*=4 */,
	MarkPointColorType color_type /*= BlackDownWhiteUp*/, CodePointBitesType code_bites_type /*=CodeBites15*/, double thread_value/* = 0*/)
{
	// 	if (center_x>679&&center_x<721&&center_y>568&&center_y<590)
	// 	{
	// 		int a =0;
	// 	}
	//cv::Mat proce_image;
	//cv::cvtColor(image_mat, proce_image, cv::COLOR_GRAY2BGR);
	float ellipse_a2 = ellipse_a * ((ratio_k2 + ratio_k1) / 2.0 - (ratio_k2 - ratio_k1) / 2.0 * 0.75);
	float ellipse_b2 = ellipse_b * ((ratio_k2 + ratio_k1) / 2.0 - (ratio_k2 - ratio_k1) / 2.0 * 0.75);
	float ellipse_a4 = ellipse_a * ((ratio_k2 + ratio_k1) / 2.0 + (ratio_k2 - ratio_k1) / 2.0 * 0.75);
	float ellipse_b4 = ellipse_b * ((ratio_k2 + ratio_k1) / 2.0 + (ratio_k2 - ratio_k1) / 2.0 * 0.75);

	int i_top = 0;
	int i_bottom = 0;
	int j_left = 0;
	int j_right = 0;
	if (int(center_y - ellipse_b4) < 0)
	{
		return false;
	}
	else i_top = int(center_y - ellipse_b4);
	if (int(center_y + ellipse_b4) > image_mat.rows - 1)
	{
		return false;
	}
	else i_bottom = int(center_y + ellipse_b4);
	if (int(center_x - ellipse_b4) < 0)
	{
		return false;
	}
	else j_left = int(center_x - ellipse_b4);
	if (int(center_x + ellipse_b4) > image_mat.cols - 1)
	{
		return false;
	}
	else j_right = int(center_x + ellipse_b4);

	int coed_bites_num;
	switch (code_bites_type)
	{
	case CodeBites15:
		coed_bites_num = 15;
		break;
	case CodeBites12:
		coed_bites_num = 12;
		break;
	}

	QList<QPointF> ellipse_ring_points;
	QList<QPointF> ellipse_ring_points_for_cal_1;
	QList<QPointF> ellipse_ring_points_for_cal_2;
	QList<int> ellipse_ring_gray_value;

	//将圆环分成15段，每段取20个点的值
	int per_num_point = 30;

	for (int i = 0; i < coed_bites_num * per_num_point; i++)
	{
		float x = cos(float(i) / coed_bites_num / per_num_point * 2 * M_PI);
		float y = sin(float(i) / coed_bites_num / per_num_point * 2 * M_PI);

		float ell_cor_r2_x = ellipse_a2 * x;
		float ell_cor_r2_y = ellipse_b2 * y;
		float ell_cor_r3_x = ellipse_a4 * x;
		float ell_cor_r3_y = ellipse_b4 * y;



		float car_cor_r2_x = ell_cor_r2_x * cos(angle_in_pi) - ell_cor_r2_y * sin(angle_in_pi) + center_x;
		float car_cor_r2_y = ell_cor_r2_x * sin(angle_in_pi) + ell_cor_r2_y * cos(angle_in_pi) + center_y;
		float car_cor_r3_x = ell_cor_r3_x * cos(angle_in_pi) - ell_cor_r3_y * sin(angle_in_pi) + center_x;
		float car_cor_r3_y = ell_cor_r3_x * sin(angle_in_pi) + ell_cor_r3_y * cos(angle_in_pi) + center_y;
		QPoint point1 = QPoint(int(car_cor_r2_x), int(car_cor_r2_y));
		QPoint point2 = QPoint(int(car_cor_r3_x), int(car_cor_r3_y));
		QList<int> gray_list = GetALineGrayList(image_mat, point1, point2);
		int mid_value = MIdValue(gray_list);
		ellipse_ring_points_for_cal_1.append(QPointF(car_cor_r2_x, car_cor_r2_y));
		ellipse_ring_points_for_cal_2.append(QPointF(car_cor_r3_x, car_cor_r3_y));
		ellipse_ring_points.append(QPointF(x, y));
		ellipse_ring_gray_value.append(mid_value);
	}
	//proce_image = proce_image(cv::Range(i_top, i_bottom), cv::Range(j_left, j_right));
	QList<int> ellipse_ring_gray_value_new;
	for (int ii = 2; ii < ellipse_ring_gray_value.size() - 2; ii++)
	{
		QList<int> mid_list = { ellipse_ring_gray_value[ii - 2],ellipse_ring_gray_value[ii - 1], ellipse_ring_gray_value[ii], ellipse_ring_gray_value[ii + 2],
		ellipse_ring_gray_value[ii + 2] };
		int mid_value_list = MIdValue(mid_list);
		if (ii == 2)
		{
			ellipse_ring_gray_value_new.append(mid_value_list);
			ellipse_ring_gray_value_new.append(mid_value_list);
			ellipse_ring_gray_value_new.append(mid_value_list);
		}
		else if (ii == ellipse_ring_gray_value.size() - 3)
		{
			ellipse_ring_gray_value_new.append(mid_value_list);
			ellipse_ring_gray_value_new.append(mid_value_list);
			ellipse_ring_gray_value_new.append(mid_value_list);
		}
		else
		{
			ellipse_ring_gray_value_new.append(mid_value_list);
		}
	}
	ellipse_ring_gray_value = ellipse_ring_gray_value_new;
	////找单位圆编码的边界，一维卷积模板半宽为3
	int half_length = per_num_point;
	int edge_index = 0;
	float ans_max = 0;
	for (int i = 0; i < ellipse_ring_points.size(); i++)
	{
		float ans = 0;
		for (int j = 0; j < half_length; j++)
		{
			int index1 = i + j;
			if (index1 >= ellipse_ring_gray_value.size())
			{
				index1 = index1 % ellipse_ring_gray_value.size();
			}
			int index2 = i + j + half_length;
			if (index2 >= ellipse_ring_gray_value.size())
			{
				index2 = index2 % ellipse_ring_gray_value.size();
			}
			ans += -ellipse_ring_gray_value.at(index1) + ellipse_ring_gray_value.at(index2);
		}
		if (abs(ans) > ans_max)
		{
			ans_max = abs(ans);
			edge_index = i + half_length;
		}
	}

	if (ans_max / half_length < 20)
	{
		return false;
	}
	//////////解编码，24度顺时针

	QList<int> code_in_2;
	double threshold_value = 0 /*= 150*/;       //阈值,要修改
	for (int i = edge_index - half_length; i < edge_index + half_length; i++)
	{
		int index1 = i;
		if (index1 >= ellipse_ring_gray_value.size())
		{
			index1 = index1 % ellipse_ring_gray_value.size();
		}
		threshold_value += ellipse_ring_gray_value[index1];
	}
	threshold_value /= half_length * 2;
	double bit_0_value = 0;
	double bit_1_value = 0;
	int bit_0_number = 0;
	int bit_1_number = 0;
	for (int i = 0; i < coed_bites_num; i++)
	{
		double mean_gray = 0;
		int num = 0;
		for (int j = 0; j < per_num_point; j++)
		{
			int index = edge_index + i * per_num_point + j + 1;
			if (index >= ellipse_ring_gray_value.size())
			{
				index = index % ellipse_ring_gray_value.size();
			}
			mean_gray += ellipse_ring_gray_value[index];
		}
		mean_gray /= per_num_point;

		if (color_type == BlackDownWhiteUp)
		{
			if (mean_gray > threshold_value)
			{
				bit_1_value += mean_gray;
				bit_1_number++;
				code_in_2.append(1);
			}
			else
			{
				bit_0_value += mean_gray;
				bit_0_number++;
				code_in_2.append(0);
			}
		}
		else
		{
			if (mean_gray > threshold_value)
			{
				bit_0_value += mean_gray;
				bit_0_number++;
				code_in_2.append(0);
			}
			else
			{
				bit_1_value += mean_gray;
				bit_1_number++;
				code_in_2.append(1);
			}
		}
	}
	bit_1_value /= (double)bit_1_number;
	bit_0_value /= (double)bit_0_number;
	if (abs(bit_1_value- bit_0_value) < thread_value)
	{
		return false;
	}
	int code_id;
	QList<int> code_out_2;
	CalculateRealCodeID20140210(code_in_2, code_out_2, code_id);

	out_put_code_id = code_id;

	return true;
}

bool ImageDetectMethod::CalculateRealCodeID20140210(QList<int> in_put_code_list, QList<int>& out_put_code_list, int& out_put_code_ID)
{
	out_put_code_ID = Change2To10(in_put_code_list);
	out_put_code_list = in_put_code_list;

	int n = in_put_code_list.size();

	for (int i = 1; i <= n - 1; i++)
	{
		QList<int> new_code_list;
		int new_id;
		for (int j = 0; j < n; j++)
		{
			if (i + j <= n - 1)
			{
				new_code_list.append(in_put_code_list.at(i + j));
			}
			else
			{
				new_code_list.append(in_put_code_list.at(i + j - n));
			}
		}
		new_id = Change2To10(new_code_list);
		if (out_put_code_ID > new_id)
		{
			out_put_code_ID = new_id;
			out_put_code_list = new_code_list;
		}
	}
	return true;
}
MarkPointColorType ImageDetectMethod::JudgeTargetColorType(const cv::Mat& sub_mat, float center_x_insubmat, float center_y_insubmat,
	float ellipse_a, float ellipse_b, float angle_in_pi)
{
	float target_gray = 0;
	float back_ground_gray = 0;
	int target_pixel_num = 0;
	int back_ground_pixel_num = 0;
	for (int i = 0; i < sub_mat.rows; i++)
	{
		const uchar* _sub_mat_ptr = sub_mat.ptr<uchar>(i);
		for (int j = 0; j < sub_mat.cols; j++)
		{
			//坐标转换图像坐标系转椭圆坐标系
			float tr_x = (j - center_x_insubmat) * cos(angle_in_pi) + (i - center_y_insubmat) * sin(angle_in_pi);
			float tr_y = -(j - center_x_insubmat) * sin(angle_in_pi) + (i - center_y_insubmat) * cos(angle_in_pi);
			if (tr_x * tr_x / ellipse_a / ellipse_a + tr_y * tr_y / ellipse_b / ellipse_b < 1)
			{
				target_gray += _sub_mat_ptr[j];
				target_pixel_num++;
			}
			else
			{
				back_ground_gray += _sub_mat_ptr[j];
				back_ground_pixel_num++;
			}
		}
	}
	target_gray /= target_pixel_num;
	back_ground_gray /= back_ground_pixel_num;

	float gray_thresh = (target_gray + back_ground_gray) / 2;

	if (target_gray > back_ground_gray)
	{
		return BlackDownWhiteUp;
	}
	else
		return WhiteDownBlackUp;
}

bool ImageDetectMethod::UncodePointCheck(const cv::Mat& image_mat, float center_x, float center_y, float ellipse_a, float ellipse_b,
	float angle_in_pi, float ratio_k /*=2*/,
	MarkPointColorType color_type /*= BlackDownWhiteUp*/, CodePointBitesType code_bites_type /*=CodeBites15*/)//非编码点的判断，c
{
	double ellipse_a1 = ellipse_a * 1.1;
	double ellipse_b1 = ellipse_b * 1.1;
	double ellipse_a2 = ellipse_a * (ratio_k - 0.1);
	double ellipse_b2 = ellipse_b * (ratio_k - 0.1);

	int i_top = 0;
	int i_bottom = 0;
	int j_left = 0;
	int j_right = 0;
	if (int(center_y - ellipse_b2) < 0)
	{
		return false;
	}
	else i_top = int(center_y - ellipse_b2);
	if (int(center_y + ellipse_b2) > image_mat.rows - 1)
	{
		return false;
	}
	else i_bottom = int(center_y + ellipse_b2);
	if (int(center_x - ellipse_b2) < 0)
	{
		return false;
	}
	else j_left = int(center_x - ellipse_b2);
	if (int(center_x + ellipse_b2) > image_mat.cols - 1)
	{
		return false;
	}
	else j_right = int(center_x + ellipse_b2);

	int coed_bites_num;
	switch (code_bites_type)
	{
	case CodeBites15:
		coed_bites_num = 15;
		break;
	case CodeBites12:
		coed_bites_num = 12;
		break;
	}

	QList<QPointF> ellipse_ring_points;
	QList<int> ellipse_ring_gray_value;

	//将圆环分成15段，每段取20个点的值
	int per_num_point = 10;

	for (int i = 0; i < coed_bites_num * per_num_point; i++)
	{
		float x = cos(float(i) / coed_bites_num / per_num_point * 2 * M_PI);
		float y = sin(float(i) / coed_bites_num / per_num_point * 2 * M_PI);

		float ell_cor_r1_x = ellipse_a1 * x;
		float ell_cor_r1_y = ellipse_b1 * y;
		float ell_cor_r2_x = ellipse_a2 * x;
		float ell_cor_r2_y = ellipse_b2 * y;

		float car_cor_r1_x = ell_cor_r1_x * cos(angle_in_pi) - ell_cor_r1_y * sin(angle_in_pi) + center_x;
		float car_cor_r1_y = ell_cor_r1_x * sin(angle_in_pi) + ell_cor_r1_y * cos(angle_in_pi) + center_y;
		float car_cor_r2_x = ell_cor_r2_x * cos(angle_in_pi) - ell_cor_r2_y * sin(angle_in_pi) + center_x;
		float car_cor_r2_y = ell_cor_r2_x * sin(angle_in_pi) + ell_cor_r2_y * cos(angle_in_pi) + center_y;

		QPoint point1 = QPoint(int(car_cor_r1_x), int(car_cor_r1_y));
		QPoint point2 = QPoint(int(car_cor_r2_x), int(car_cor_r2_y));

		QList<int> gray_list = GetALineGrayList(image_mat, point1, point2);
		//int mid_value = MIdValue(gray_list);
		int mid_value = AverageOfList(gray_list);

		ellipse_ring_points.append(QPointF(x, y));
		ellipse_ring_gray_value.append(mid_value);
	}

	////找单位圆编码的边界，一维卷积模板半宽为3
	int half_length = 5;
	int edge_index = 0;
	float ans_max = 0;
	for (int i = 0; i < ellipse_ring_points.size() - 2 * half_length; i++)
	{
		float ans = 0;
		for (int j = 0; j < half_length; j++)
		{
			ans += -ellipse_ring_gray_value.at(i + j) + ellipse_ring_gray_value.at(i + j + half_length);
		}
		if (ans > ans_max)
		{
			ans_max = ans;
			edge_index = i + half_length;
		}
	}

	//阈值
	int delta_M = 20;
	if (ans_max / half_length > delta_M)
	{
		return false;
	}
	else
		return true;
}
bool ImageDetectMethod::FindSubPixelPosOfCircleCenter(const cv::Mat& image_mat, 
	float center_x, float center_y, float ellipse_a, float ellipse_b,
	float angle_in_pi, const std::vector<cv::Point>& contour_points,
	float& sub_pixel_center_x, float& sub_pixel_center_y,
	std::vector<cv::Point2f>& subpixel_edge_points /*= NULL*/, SubPixelPosMethod subpixel_pos_method)
{
	switch (subpixel_pos_method)
	{
	case NoSubPixel_Match:
	{
		sub_pixel_center_x = center_x;
		sub_pixel_center_y = center_y;

		if (contour_points.size() != 0)
		{
			subpixel_edge_points.clear();
			for (int i = 0; i < contour_points.size(); i++)
			{
				subpixel_edge_points.push_back(cv::Point2f(contour_points.at(i).x, contour_points.at(i).y));
			}
		}
	}
	break;
	case SubPixel_Interpolation:
	{
		float ellipse_a2 = ellipse_a * 1.5;
		float ellipse_b2 = ellipse_b * 1.5;

		int i_top = 0;
		int i_bottom = 0;
		int j_left = 0;
		int j_right = 0;
		if (int(center_y - ellipse_b2) < 0)
		{
			i_top = 0;
		}
		else i_top = int(center_y - ellipse_b2);
		if (int(center_y + ellipse_b2) > image_mat.rows - 1)
		{
			i_bottom = image_mat.rows - 1;
		}
		else i_bottom = int(center_y + ellipse_b2);
		if (int(center_x - ellipse_b2) < 0)
		{
			j_left = 0;
		}
		else j_left = int(center_x - ellipse_b2);
		if (int(center_x + ellipse_b2) > image_mat.cols - 1)
		{
			j_right = image_mat.cols - 1;
		}
		else j_right = int(center_x + ellipse_b2);

		cv::Mat crop_image = image_mat(cv::Range(i_top, i_bottom), cv::Range(j_left, j_right));
		std::vector< std::vector<cv::Point2f>> sub_contours;
		Sub_pixel_edge(crop_image, sub_contours);
		float sub_x_update, sub_y_update;
		float min_value = 1e20;
		if (sub_contours.size() == 0)
		{
			return false;
		}
		sub_pixel_center_x = center_x;
		sub_pixel_center_y = center_y;
		for (int ii = 0; ii < sub_contours.size(); ii++)
		{
			std::vector<cv::Point2f> con_now;
			for (int jj = 0; jj < sub_contours[ii].size(); jj++)
			{
				con_now.push_back(sub_contours[ii][jj]);
			}
			if (con_now.size() < 5)
			{
				continue;
			}
			cv::RotatedRect rRect = fitEllipseAMS(con_now);
			if (isnan(rRect.center.x) || isinf(rRect.center.x))
			{
				continue;
			}
			if (isnan(rRect.center.y) || isinf(rRect.center.y))
			{
				continue;
			}
			if (isnan(rRect.size.width) || isinf(rRect.size.width))
			{
				continue;
			}
			if (isnan(rRect.size.height) || isinf(rRect.size.height))
			{
				continue;
			}
			if (abs((float)i_top + rRect.center.y - center_y) + abs((float)j_left + rRect.center.x - center_x) < min_value)
			{
				subpixel_edge_points.clear();
				for (int ss = 0; ss < sub_contours[ii].size(); ss++)
				{
					subpixel_edge_points.push_back(cv::Point2f(sub_contours[ii][ss].x + (float)j_left
						, sub_contours[ii][ss].y + (float)i_top));
				}
				min_value = abs((float)i_top + rRect.center.y - center_y) + abs((float)j_left + rRect.center.x - center_x);
				sub_x_update = rRect.center.x;
				sub_y_update = rRect.center.y;
				sub_pixel_center_y = (float)i_top + sub_y_update;
				sub_pixel_center_x = (float)j_left + sub_x_update;
			}
		}
	}
	break;
	case Gray_Moment:
	{
		///灰度矩方法，一维连续函数前3阶灰度矩
		std::vector<cv::Point2f> new_contour_points;

		for (int n = 0; n < contour_points.size(); n++)
		{
			int y = contour_points[n].y;
			int x = contour_points[n].x;

			if (x - 3 < 0 || x + 3 > image_mat.cols - 1 || y - 3 < 0 || y + 3 > image_mat.rows - 1)
			{
				continue;
			}

			if (x - center_x == 0)
			{
				continue;
			}

			float sita = atan2(y - center_y, x - center_x);
			float k = tan(sita);
			//float k = (y-center_y)/(x-center_x);   //梯度方向
			// 			double sita = gray_gradient_sita.at<double>(y,x);
			// 			double k = tan(sita);
			//double f_a,f_b,f_c,f_d;
			float abs_k = abs(k);

			float positive_gray[3];
			float negative_gray[3];

			int invt = 1;
			if (k > 0)
			{
				invt = -1;
			}
			if (abs_k < 1)
			{
				for (int i = 1; i < 4; i++)
				{
					int a = int(i * abs_k) / 1;
					float lamd2 = i * abs_k - a;
					float lamd1 = 1 - lamd2;

					positive_gray[i - 1] = lamd1 * image_mat.at<uchar>(y - a, x + invt * i) + lamd2 * image_mat.at<uchar>(y - a - 1, x + invt * i);
					negative_gray[i - 1] = lamd1 * image_mat.at<uchar>(y + a, x - invt * i) + lamd2 * image_mat.at<uchar>(y + a + 1, x - invt * i);
				}
			}
			else
			{
				for (int i = 1; i < 4; i++)
				{
					int a = int(i / abs_k) / 1;
					float lamd2 = i / abs_k - a;
					float lamd1 = 1 - lamd2;

					positive_gray[i - 1] = lamd1 * image_mat.at<uchar>(y - i, x + invt * a) + lamd2 * image_mat.at<uchar>(y - i, x + invt * (a + 1));
					negative_gray[i - 1] = lamd1 * image_mat.at<uchar>(y + i, x - invt * a) + lamd2 * image_mat.at<uchar>(y + i, x - invt * (a + 1));
				}
			}

			float gray_value[7];    //沿灰度梯度方向 ，向外
			if (y < center_y)
			{
				gray_value[0] = negative_gray[2];
				gray_value[1] = negative_gray[1];
				gray_value[2] = negative_gray[0];
				gray_value[3] = image_mat.at<uchar>(y, x);
				gray_value[4] = positive_gray[0];
				gray_value[5] = positive_gray[1];
				gray_value[6] = positive_gray[2];
			}
			else
			{
				gray_value[0] = positive_gray[2];
				gray_value[1] = positive_gray[1];
				gray_value[2] = positive_gray[0];
				gray_value[3] = image_mat.at<uchar>(y, x);
				gray_value[4] = negative_gray[0];
				gray_value[5] = negative_gray[1];
				gray_value[6] = negative_gray[2];
			}

			if (gray_value[0] > gray_value[6])
			{
				std::swap(gray_value[0], gray_value[6]);
				std::swap(gray_value[1], gray_value[5]);
				std::swap(gray_value[2], gray_value[4]);
				sita = sita + M_PI;
			}

			////计算灰度矩
			int N = 7;
			float m1 = 0, m2 = 0, m3 = 0;
			for (int i = 0; i < N; i++)
			{
				m1 += gray_value[i];
				m2 += gray_value[i] * gray_value[i];
				m3 += gray_value[i] * gray_value[i] * gray_value[i];
			}
			m1 /= N;
			m2 /= N;
			m3 /= N;

			float sigm = sqrt(m2 - m1 * m1);
			float s = (m3 + 2 * m1 * m1 * m1 - 3 * m1 * m2) / (sigm * sigm * sigm);

			float delta = N * 0.5 * s * sqrt(1.0 / (4 + s * s)) + (N + 1) * 0.5 - (N / 2 + 1);
			if (_isnan(delta))
			{
				continue;
			}

			new_contour_points.push_back(cv::Point2f(x + delta * cos(sita), y + delta * sin(sita)));
		}
		cv::RotatedRect new_rRect = fitEllipse(new_contour_points);    //fitEllipse只接受float和int类型

		//尝试通过拟合误差剔除坏点
		ReduceBadEllipseFitPoints(new_contour_points, new_rRect.center.x, new_rRect.center.y,
			new_rRect.size.width * 0.5, new_rRect.size.height * 0.5, new_rRect.angle * M_PI / 180);
		new_rRect = fitEllipse(new_contour_points);

		sub_pixel_center_x = new_rRect.center.x;
		sub_pixel_center_y = new_rRect.center.y;
		if (contour_points.size() != 0)
		{
			subpixel_edge_points.clear();
			for (int i = 0; i < contour_points.size(); i++)
			{
				subpixel_edge_points.push_back(cv::Point2f(contour_points.at(i).x, contour_points.at(i).y));
			}
		}

	}
		break;
	case Binary_Centroid:
	case Squared_Gray_Centroid:
	case Gray_Centroid:
	{
		QRect sub_rect = GetEllipseROIRect(image_mat, center_x, center_y, ellipse_a, ellipse_b, angle_in_pi);
		cv::Mat sub_mat = image_mat.operator ()(cv::Rect(sub_rect.x(), sub_rect.y(), sub_rect.width() + 1, sub_rect.height() + 1));

		auto color_type = JudgeTargetColorType(sub_mat, center_x - sub_rect.x(), center_y - sub_rect.y(),
			ellipse_a, ellipse_b, angle_in_pi);
		float gray_threshold;
		cv::Mat binary_image;
		gray_threshold = cv::threshold(sub_mat, binary_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		int k = 0;
		int d_gray = 6;
		float sub_x, sub_y;
		sub_x = sub_y = 0;
		for (int i = -k; i <= k; i++)
		{
			float threshold = gray_threshold + i * d_gray;
			float part_sub_x, part_sub_y;
			CalCentriodBySubsetMat(sub_mat, threshold, part_sub_x, part_sub_y, color_type, subpixel_pos_method);
			sub_x += part_sub_x;
			sub_y += part_sub_y;
		}
		sub_x /= 2 * k + 1;
		sub_y /= 2 * k + 1;

		sub_pixel_center_x = sub_x + sub_rect.x();
		sub_pixel_center_y = sub_y + sub_rect.y();

		if (contour_points.size() != 0)
		{
			subpixel_edge_points.clear();
			for (int i = 0; i < contour_points.size(); i++)
			{
				subpixel_edge_points.push_back(cv::Point2f(contour_points.at(i).x, contour_points.at(i).y));
			}
		}
	}
		break;
	default:
	{
		sub_pixel_center_x = center_x;
		sub_pixel_center_y = center_y;

		if (contour_points.size() != 0)
		{
			subpixel_edge_points.clear();
			for (int i = 0; i < contour_points.size(); i++)
			{
				subpixel_edge_points.push_back(cv::Point2f(contour_points.at(i).x, contour_points.at(i).y));
			}
		}
	}
		break;
	}
	return true;
	
	

}
void ImageDetectMethod::ReduceBadEllipseFitPoints(std::vector<cv::Point2f>& edge_points, float center_x, float center_y,
	float ellipse_a, float ellipse_b, float angle_in_pi)
{
	//尝试通过拟合误差剔除坏点
	std::vector<float> error_vector;
	for (int i = 0; i < edge_points.size(); i++)
	{
		float error = ErrorDROfEllipseFit(center_x, center_y, ellipse_a, ellipse_b, angle_in_pi,
			edge_points[i].x, edge_points[i].y);

		error_vector.push_back(error);

		if (error > 0.5)
		{
			std::vector<cv::Point2f>::iterator it = edge_points.begin() + i;
			edge_points.erase(it);

			i--;
		}
	}
}


float ImageDetectMethod::CalWeightOfCentriod(float gray_value, float gray_threshold,
	MarkPointColorType color_type /*= BlackDownWhiteUp*/,
	SubPixelPosMethod subPixel_method /*= Gray_Centroid*/)
{
	//gray_threshold = 100;
	float weight_value;
	switch (color_type)
	{
	case BlackDownWhiteUp:
	{
		switch (subPixel_method)
		{
		case Binary_Centroid:
			if (gray_value - gray_threshold > 0)
			{
				weight_value = 1.;
			}
			else
				weight_value = 0;
			break;

		case Gray_Centroid:
			if (gray_value - gray_threshold > 0)
			{
				weight_value = gray_value /*- gray_threshold*/;
			}
			else
				weight_value = 0;

			//真实重心法
			//weight_value = gray_value;
			break;

		case Squared_Gray_Centroid:
			if (gray_value - gray_threshold > 0)
			{
				weight_value = (gray_value /*- gray_threshold*/) * (gray_value /*- gray_threshold*/);
			}
			else
				weight_value = 0;

			//真实灰度平方重心法5
			//weight_value = gray_value*gray_value;
			break;
		}
		break;
	}

	case WhiteDownBlackUp:
	{
		switch (subPixel_method)
		{
		case Binary_Centroid:
			if (gray_threshold - gray_value > 0)
			{
				weight_value = 1.;
			}
			else
				weight_value = 0;
			break;

		case Gray_Centroid:
			if (gray_threshold - gray_value > 0)
			{
				weight_value = gray_threshold - gray_value;
			}
			else
				weight_value = 0;
			break;

		case Squared_Gray_Centroid:
			if (gray_threshold - gray_value > 0)
			{
				weight_value = (gray_threshold - gray_value) * (gray_threshold - gray_value);
			}
			else
				weight_value = 0;
			break;
		}
		break;
	}
	}

	return weight_value;
}

bool ImageDetectMethod::CalCentriodBySubsetMat(const cv::Mat& subset_mat, float gray_threshold, float& sub_center_x, float& sub_center_y,
	MarkPointColorType color_type /*= BlackDownWhiteUp*/,
	SubPixelPosMethod subPixel_method /*= Gray_Centroid*/)
{
	sub_center_x = 0;
	sub_center_y = 0;
	float all_weight = 0;

	for (int i = 0; i < subset_mat.rows; i++)
	{
		const uchar* _subset_mat = subset_mat.ptr<uchar>(i);
		for (int j = 0; j < subset_mat.cols; j++)
		{
			float weight_value = CalWeightOfCentriod(float(_subset_mat[j]), gray_threshold, color_type, subPixel_method);
			sub_center_x += j * weight_value;
			sub_center_y += i * weight_value;
			all_weight += weight_value;
		}
	}

	if (all_weight == 0)
	{
		sub_center_x = 0;
		sub_center_y = 0;
	}
	else
	{
		sub_center_x /= all_weight;
		sub_center_y /= all_weight;
	}

	return true;
}
QRect ImageDetectMethod::GetEllipseROIRect(const cv::Mat& image_mat, float center_x, float center_y, float ellipse_a, float ellipse_b, float angle_in_pi)
{
	int delta = 0;

	int i_top = int(center_y - ellipse_b - delta);
	if (i_top < 0)
	{
		i_top = 0;
	}

	int i_bottom = ceil(center_y + ellipse_b + delta);
	if (i_bottom > image_mat.rows - 1)
	{
		i_bottom = image_mat.rows - 1;
	}
	int j_left = int(center_x - ellipse_b - delta);
	if (j_left < 0)
	{
		j_left = 0;
	}

	int j_right = ceil(center_x + ellipse_b + delta);
	if (j_right > image_mat.cols - 1)
	{
		j_right = image_mat.cols - 1;
	}

	return QRect(j_left, i_top, j_right - j_left, i_bottom - i_top);
}




int ImageDetectMethod::AverageOfList(QList<int>& list_value)
{
	int num = 0;
	int aver = 0;
	for (int i = 0; i < list_value.size(); i++)
	{
		aver += list_value[i];
		num++;
	}
	aver /= num;
	return aver;
}

QList<int> ImageDetectMethod::GetALineGrayList(cv::Mat image_mat, QPoint point1, QPoint point2)
{
	QList<int> gray_list;

	if (point1.x() == point2.x())
	{
		int j = point1.x();
		if (point1.y() < point2.y())
		{
			for (int i = point1.y(); i < point2.y(); i++)
			{
				gray_list.append(image_mat.at<uchar>(i, j));
			}
		}
		else
		{
			for (int i = point2.y(); i < point1.y(); i++)
			{
				gray_list.append(image_mat.at<uchar>(i, j));
			}
		}
	}
	else
	{
		double k = double(point2.y() - point1.y()) / double(point2.x() - point1.x());
		if (abs(k) <= 1)
		{
			if (point1.x() < point2.x())
			{
				for (int j = point1.x(); j <= point2.x(); j++)
				{
					int i = int(k * (j - point1.x()) + point1.y());
					gray_list.append(image_mat.at<uchar>(i, j));
				}
			}
			else
			{
				// 				for (int j = point2.x();j<point1.x();j++)
				// 				{
				// 					int i = int(k*(j-point1.x())+point1.y());
				// 					gray_list.append(image_mat.at<uchar>(i,j));
				// 				}

				for (int j = point1.x(); j >= point2.x(); j--)
				{
					int i = int(k * (j - point1.x()) + point1.y());
					gray_list.append(image_mat.at<uchar>(i, j));
				}
			}
		}
		else
		{
			k = double(point2.x() - point1.x()) / double(point2.y() - point1.y());
			if (point1.y() < point2.y())
			{
				for (int i = point1.y(); i <= point2.y(); i++)
				{
					int j = int(k * (i - point1.y()) + point1.x());
					gray_list.append(image_mat.at<uchar>(i, j));
				}
			}
			else
			{
				// 				for (int i = point2.y();i<=point1.y();i++)
				// 				{
				// 					int j = int(k*(i-point1.y())+point1.x());
				// 					gray_list.append(image_mat.at<uchar>(i,j));
				// 				}
				for (int i = point1.y(); i >= point2.y(); i--)
				{
					int j = int(k * (i - point1.y()) + point1.x());
					gray_list.append(image_mat.at<uchar>(i, j));
				}
			}
		}
	}
	return gray_list;
}
double ImageDetectMethod::MeanValue(QList<float> value_list)
{
	double mean_v = 0;
	double mean_vn = 0;
	for (int ii =0;ii< value_list.size();ii++)
	{
		mean_v += value_list[ii];
		mean_vn += 1;
	}
	if (mean_vn)
	{
		mean_v /= mean_vn;
	}
	return mean_v;
}
int ImageDetectMethod::MIdValue(QList<int> value_list)
{
	std::sort(value_list.begin(), value_list.end());

	return value_list[value_list.size() / 2];
}
int ImageDetectMethod::Change2To10(QList<int> list_code2)
{
	int ans10 = 0;
	int n = list_code2.size();
	for (int i = 0; i < n; i++)
	{
		ans10 = ans10 + int(pow(2.0, n - i - 1) * list_code2.value(i));
	}
	return ans10;
}

bool ImageDetectMethod::FindCircleGrid(cv::Mat ori_image, int h_num, int v_num,
	int h_offset, int v_offset,
	int h_mid_length, int v_mid_length,
	std::vector<cv::Point2f>& corners, std::vector<uchar>& sign_list,
	int& useful_corner_num,
	std::vector<std::vector<cv::Point2f>>& edge_points /*= std::vector<vector<cv::Point2f>>()*/,
	float max_ratio, float ratio, float min_radius/* = 5*/, float max_radius /*= 50*/, float ellipse_error_pixel /*=0.5*/, int min_arc,
	int min_points, int min_contour_num,
	DetectContoursMethod image_process_method /*= OTSU_Method*/,
	SubPixelPosMethod subpixel_pos_method /*=Gray_Centroid*/)
{
	if (!ori_image.data)        // 判断图片调入是否成功
		return false;        // 调入图片失败则退出

	int useful_corner_num_max = 0;
	std::vector<cv::Point2f> corners_max;
	std::vector<std::vector<cv::Point2f>> edge_points_max;
	cv::Mat processed_image_mat;
	std::vector<std::vector<cv::Point>> contours;
	QList<QList<float>> ellipse_pars;
	std::vector<std::vector<cv::Point>>contours_for_key;
	contours_for_key.resize(h_num * v_num);
	QList<QList<float>> ellipse_pars_ori;
	//1.图像预处理
	processed_image_mat = ori_image.clone();
	ImagePreprocess(ori_image, processed_image_mat);

	//2.边缘检测，存入闭合轮廓
	DetectClosedContours(processed_image_mat, contours, image_process_method);

	//3.轮廓筛选，尺寸，形状等准则，圆
	FilterEllipseContours(contours, min_radius, max_radius,
		ellipse_error_pixel, ellipse_pars, max_ratio, min_points, min_contour_num, min_arc);
	QList<QList<float>> ellipse_pars_copy = ellipse_pars;
	FilterEllipseContours_for_distance(contours, ellipse_pars);
	if (ellipse_pars.size() < h_num * v_num * 0.75)
	{
		return false;
	}

	FilterEllipseContoursForCSICalibrationPlane(processed_image_mat,
		ellipse_pars_copy, ellipse_pars, ellipse_pars_ori, ratio);

	std::vector<std::vector<cv::Point>> contours_copy = contours;
	contours.clear();
	for (int ii = 0; ii < contours_copy.size(); ii++)
	{
		bool need_con_c = false;
		for (int pp = 0; pp < ellipse_pars.size(); pp++)
		{
			if (ii == ellipse_pars[pp][6])
			{
				need_con_c = true;
				break;
			}
		}
		if (need_con_c)
		{
			contours.push_back(contours_copy[ii]);
		}
	}
	//cv::Mat II = cv::Mat::zeros(ori_image.size(), CV_8UC3);
	//cv::cvtColor(ori_image, II, CV_GRAY2BGR);
	////cv::drawContours(II, contours_copy, -1, cv::Scalar(0, 0, 255), 10);
	//cv::drawContours(II, contours, -1, cv::Scalar(0, 255, 0), 5);
	//cv::namedWindow("canny", cv::WINDOW_NORMAL);
	//cv::imshow("canny", II);
	//cv::waitKey(0);
	std::vector<cv::Point2f> points_temp;
	corners.clear();
	for (int ii = 0; ii < ellipse_pars.size(); ii++)
	{
		points_temp.push_back(cv::Point2f(ellipse_pars[ii][0], ellipse_pars[ii][1]));
	}

	if (ellipse_pars_ori.size() > 50)
	{
		return false;
	}
	for (int kk = 0; kk < ellipse_pars_ori.size() - 2; kk++)
	{
		for (int pp = kk + 1; pp < ellipse_pars_ori.size() - 1; pp++)
		{
			for (int qq = pp + 1; qq < ellipse_pars_ori.size(); qq++)
			{
				std::vector<int> orident_point_index_list;
				orident_point_index_list.push_back(kk);
				orident_point_index_list.push_back(pp);
				orident_point_index_list.push_back(qq);
				int original_point_index = 0, X_axis_point_index = 0, Y_axis_point_index = 0;
				float d_max = 0;
				for (int i = 0; i < orident_point_index_list.size() - 1; i++)
				{
					for (int j = i + 1; j < orident_point_index_list.size(); j++)
					{
						float x1 = ellipse_pars_ori[orident_point_index_list[i]][0];
						float y1 = ellipse_pars_ori[orident_point_index_list[i]][1];
						float x2 = ellipse_pars_ori[orident_point_index_list[j]][0];
						float y2 = ellipse_pars_ori[orident_point_index_list[j]][1];
						float length = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
						if (length > d_max)
						{
							d_max = length;
							X_axis_point_index = orident_point_index_list[i];
							Y_axis_point_index = orident_point_index_list[j];
						}
					}
				}
				for (int i = 0; i < orident_point_index_list.size(); i++)
				{
					if (orident_point_index_list[i] != X_axis_point_index && orident_point_index_list[i] != Y_axis_point_index)
					{
						original_point_index = orident_point_index_list[i];
					}
				}

				//***判断两个直角边上两方向点之间的圆点个数，确定正确的XY轴
				std::vector<float> line_X_points_length;
				std::vector<float> line_Y_points_length;
				std::vector<cv::Point2f> line_X_points;
				std::vector<cv::Point2f> line_Y_points;
				cv::Point2f original_point(ellipse_pars_ori[original_point_index][0], ellipse_pars_ori[original_point_index][1]);
				cv::Point2f X_axis_point(ellipse_pars_ori[X_axis_point_index][0], ellipse_pars_ori[X_axis_point_index][1]);
				cv::Point2f Y_axis_point(ellipse_pars_ori[Y_axis_point_index][0], ellipse_pars_ori[Y_axis_point_index][1]);
				float X_line_A, X_line_B, X_line_C, Y_line_A, Y_line_B, Y_line_C;
				LineEquation(original_point, X_axis_point, X_line_A, X_line_B, X_line_C);
				LineEquation(original_point, Y_axis_point, Y_line_A, Y_line_B, Y_line_C);

				double mean_ab = (sqrt(ellipse_pars_ori[original_point_index][2] * ellipse_pars_ori[original_point_index][3]) / 2.0
					+ sqrt(ellipse_pars_ori[X_axis_point_index][2] * ellipse_pars_ori[X_axis_point_index][3]) / 2.0
					+ sqrt(ellipse_pars_ori[Y_axis_point_index][2] * ellipse_pars_ori[Y_axis_point_index][3]) / 2.0) / 3.0;
				double dis_r_x = 0;
				double dis_r_y = 0;
				double L_x = sqrt(pow(original_point.x - X_axis_point.x, 2) + pow(original_point.y - X_axis_point.y, 2));
				double L_y = sqrt(pow(original_point.x - Y_axis_point.x, 2) + pow(original_point.y - Y_axis_point.y, 2));
				std::vector<int> Line_x_exist;
				std::vector<int> Line_y_exist;
				if (h_mid_length > v_mid_length)
				{
					if (L_x > L_y)
					{
						dis_r_x = L_x / (double)h_mid_length;
						dis_r_y = L_y / (double)v_mid_length;
					}
					else
					{
						dis_r_x = L_x / (double)v_mid_length;
						dis_r_y = L_y / (double)h_mid_length;
					}
				}
				else
				{
					if (L_x > L_y)
					{
						dis_r_x = L_x / (double)v_mid_length;
						dis_r_y = L_y / (double)h_mid_length;
					}
					else
					{
						dis_r_x = L_x / (double)h_mid_length;
						dis_r_y = L_y / (double)v_mid_length;
					}
				}

				for (int i = 0; i < ellipse_pars.size(); i++)
				{
					if (ellipse_pars[i][6]== ellipse_pars_ori[original_point_index][6])
					{
						continue;
					}
					if (ellipse_pars[i][6] == ellipse_pars_ori[X_axis_point_index][6])
					{
						continue;
					}
					if (ellipse_pars[i][6] == ellipse_pars_ori[Y_axis_point_index][6])
					{
						continue;
					}

					if (sqrt(ellipse_pars[i][2] * ellipse_pars[i][2]) / 2.0 < mean_ab * 0.5 || sqrt(ellipse_pars[i][2] * ellipse_pars[i][2]) / 2.0 > mean_ab * 2)
					{
						continue;
					}
					double L2ori = sqrt(pow(original_point.x - ellipse_pars[i][0], 2) + pow(original_point.y - ellipse_pars[i][1], 2));

					float d_max = ellipse_pars[i][2];

					cv::Point2f p(ellipse_pars[i][0], ellipse_pars[i][1]);
					float dis_to_X_line = PointTpLineDistance(p, X_line_A, X_line_B, X_line_C);
					float dis_to_Y_line = PointTpLineDistance(p, Y_line_A, Y_line_B, Y_line_C);

					float dot_in_X_line = (p.x - original_point.x) * (p.x - X_axis_point.x) + (p.y - original_point.y) * (p.y - X_axis_point.y);
					float dot_in_Y_line = (p.x - original_point.x) * (p.x - Y_axis_point.x) + (p.y - original_point.y) * (p.y - Y_axis_point.y);

					if (dis_to_X_line < d_max && dot_in_X_line < 0)
					{
						line_X_points_length.push_back(L2ori);
						line_X_points.push_back(p);
					}

					if (dis_to_Y_line < d_max && dot_in_Y_line < 0)
					{
						line_Y_points.push_back(p);
						line_Y_points_length.push_back(L2ori);
					}
				}
				corners.clear();
				sign_list.clear();
				corners.resize(h_num * v_num);
				edge_points.clear();
				edge_points.resize(h_num * v_num);
				for (int sc = 0; sc < h_num * v_num; sc++)
				{
					corners[sc].x = 0;
					corners[sc].y = 0;
				}
				sign_list.resize(h_num * v_num);
				useful_corner_num = 0;

				if ((line_X_points.size() == h_mid_length - 1 && line_Y_points.size() == v_mid_length - 1)
					||(line_X_points.size() == v_mid_length - 1 && line_Y_points.size() == h_mid_length - 1))
				{
					if (line_X_points.size() == v_mid_length - 1 && line_Y_points.size() == h_mid_length - 1)
					{
						std::swap(X_axis_point_index, Y_axis_point_index);
						std::swap(line_X_points, line_Y_points);
						std::swap(X_axis_point, Y_axis_point);
					}

					//右手系判断，向量叉乘大于0
					cv::Point2f vector1(X_axis_point.x - original_point.x, X_axis_point.y - original_point.y);
					cv::Point2f vector2(Y_axis_point.x - original_point.x, Y_axis_point.y - original_point.y);
					if (vector1.x * vector2.y - vector1.y * vector2.x > 0)
					{
						continue;
					}
					std::vector<cv::Point2f> src_points, dst_points;
					src_points.push_back(original_point);
					src_points.push_back(X_axis_point);
					src_points.push_back(Y_axis_point);
					dst_points.push_back(cv::Point2f(h_offset, v_offset));
					dst_points.push_back(cv::Point2f(h_offset + h_mid_length, v_offset));
					dst_points.push_back(cv::Point2f(h_offset, v_offset + v_mid_length));
					for (int i = 0; i < line_X_points.size() - 1; i++)
					{
						for (int j = 0; j < line_X_points.size() - i - 1; j++)
						{
							float d1 = PointToPointDistance(line_X_points[j], original_point);
							float d2 = PointToPointDistance(line_X_points[j + 1], original_point);
							if (d1 > d2)
							{
								std::swap(line_X_points[j], line_X_points[j + 1]);
							}
						}
					}
					for (int i = 0; i < line_Y_points.size() - 1; i++)
					{
						for (int j = 0; j < line_Y_points.size() - i - 1; j++)
						{
							float d1 = PointToPointDistance(line_Y_points[j], original_point);
							float d2 = PointToPointDistance(line_Y_points[j + 1], original_point);
							if (d1 > d2)
							{
								std::swap(line_Y_points[j], line_Y_points[j + 1]);
							}
						}
					}
					for (int i = 0; i < line_X_points.size(); i++)
					{
						src_points.push_back(line_X_points[i]);
						dst_points.push_back(cv::Point2f(h_offset + i + 1, v_offset));
					}
					for (int i = 0; i < line_Y_points.size(); i++)
					{
						src_points.push_back(line_Y_points[i]);
						dst_points.push_back(cv::Point2f(h_offset, v_offset + i + 1));
					}

					//
					cv::Mat H_mat = cv::findHomography(src_points, dst_points, cv::RANSAC);
					H_mat.convertTo(H_mat, CV_32F);

					//5.4 重新按顺序排列

					std::vector<float> coners_cost;
					coners_cost.resize(h_num * v_num);
					for (int ii = 0; ii < coners_cost.size(); ii++)
					{
						coners_cost[ii] = -1;
					}
					for (int n = 0; n < ellipse_pars.size(); n++)
					{
						cv::Mat X = cv::Mat(3, 1, CV_32F);
						X.at<float>(0, 0) = ellipse_pars[n][0];
						X.at<float>(1, 0) = ellipse_pars[n][1];
						X.at<float>(2, 0) = 1;
						cv::Mat A = H_mat * X;
						float new_x = A.at<float>(0, 0) / A.at<float>(2, 0);
						float new_y = A.at<float>(1, 0) / A.at<float>(2, 0);

						int i = floor(new_y + 0.5);
						int j = floor(new_x + 0.5);
						float delta_x = abs(new_x - j);
						float delta_y = abs(new_y - i);

						if (i<1 || i>v_num || j<1 || j>h_num)
						{
							continue;
						}
						if ((i - 1)==9 && (j - 1) == 12)
						{
							int scacas = 1;
						}
						if (coners_cost[(i - 1) * h_num + j - 1] == -1)
						{
							coners_cost[(i - 1) * h_num + j - 1] = delta_x + delta_y;
							float sub_pixel_x, sub_pixel_y;
							std::vector<cv::Point2f> edge_contour;

							if (FindSubPixelPosOfCircleCenter(processed_image_mat, ellipse_pars[n][0], ellipse_pars[n][1], ellipse_pars[n][2],
								ellipse_pars[n][3], ellipse_pars[n][4], contours_copy[ellipse_pars[n][6]], sub_pixel_x, sub_pixel_y,
								edge_contour, subpixel_pos_method))
							{
								ellipse_pars[n][0] = sub_pixel_x;
								ellipse_pars[n][1] = sub_pixel_y;

								edge_points[(i - 1) * h_num + j - 1] = edge_contour;
								corners[(i - 1) * h_num + j - 1] = cv::Point2f(sub_pixel_x, sub_pixel_y);
								sign_list[(i - 1) * h_num + j - 1] = 2;
							}
							else
							{
								for (int tp = 0; tp < contours_copy[ellipse_pars[n][6]].size(); tp++)
								{
									edge_contour.push_back(cv::Point2f(contours_copy[ellipse_pars[n][6]][tp].x,
										contours_copy[ellipse_pars[n][6]][tp].x));
								}
								edge_points[(i - 1) * h_num + j - 1] = edge_contour;
								corners[(i - 1) * h_num + j - 1] = cv::Point2f(ellipse_pars[n][0], ellipse_pars[n][1]);
								sign_list[(i - 1) * h_num + j - 1] = 1;
							}
							contours_for_key[(i - 1) * h_num + j - 1] = contours_copy[ellipse_pars[n][6]];
							sign_list[(i - 1) * h_num + j - 1] = 2;
							useful_corner_num++;
						}
						else if (coners_cost[(i - 1) * h_num + j - 1] < (delta_x + delta_y))
						{
							continue;
						}
						else
						{
							coners_cost[(i - 1) * h_num + j - 1] = delta_x + delta_y;
							float sub_pixel_x, sub_pixel_y;
							std::vector<cv::Point2f> edge_contour;
							if (FindSubPixelPosOfCircleCenter(processed_image_mat, ellipse_pars[n][0], ellipse_pars[n][1], ellipse_pars[n][2],
								ellipse_pars[n][3], ellipse_pars[n][4], contours_copy[ellipse_pars[n][6]], sub_pixel_x, sub_pixel_y,
								edge_contour, subpixel_pos_method))
							{
								ellipse_pars[n][0] = sub_pixel_x;
								ellipse_pars[n][1] = sub_pixel_y;
								edge_points[(i - 1) * h_num + j - 1] = edge_contour;
								corners[(i - 1) * h_num + j - 1] = cv::Point2f(sub_pixel_x, sub_pixel_y);
								sign_list[(i - 1) * h_num + j - 1] = 2;
							}
							else
							{
								for (int tp = 0; tp < contours_copy[ellipse_pars[n][6]].size(); tp++)
								{
									edge_contour.push_back(cv::Point2f(contours_copy[ellipse_pars[n][6]][tp].x,
										contours_copy[ellipse_pars[n][6]][tp].x));
								}
								edge_points[(i - 1) * h_num + j - 1] = edge_contour;
								corners[(i - 1) * h_num + j - 1] = cv::Point2f(ellipse_pars[n][0], ellipse_pars[n][1]);
								sign_list[(i - 1) * h_num + j - 1] = 1;
							}
							contours_for_key[(i - 1) * h_num + j - 1] = contours_copy[ellipse_pars[n][6]];
						}
					}
					if (useful_corner_num> useful_corner_num_max)
					{
						useful_corner_num_max = useful_corner_num;
						corners_max = corners;
						edge_points_max = edge_points;
					}
					if (useful_corner_num == (h_num * v_num))
					{
						return true;
					}

					//std::vector<std::vector<cv::Point>>contours_for_key_C;
					//for (int pt = 0; pt < contours_for_key.size(); pt++)
					//{
					//	if(contours_for_key[pt].size()!=0)
					//		contours_for_key_C.push_back(contours_for_key[pt]);
					//}
					//cv::Mat II = cv::Mat::zeros(ori_image.size(), CV_8UC3);
					//cv::cvtColor(ori_image, II, CV_GRAY2BGR);
					////cv::drawContours(II, contours_copy, -1, cv::Scalar(0, 0, 255), 10);
					//cv::drawContours(II, contours_for_key_C, -1, cv::Scalar(0, 255, 0), 5);
					//cv::namedWindow("cannys", cv::WINDOW_NORMAL);
					//cv::imshow("cannys", II);
					//cv::waitKey(0);
				}
			}
		}
	}

	if (useful_corner_num == (h_num * v_num))
	{
		return true;
	}
	else if (useful_corner_num_max >= (h_num * v_num) * 0.75)
	{
		useful_corner_num = useful_corner_num_max;
		corners = corners_max;
		edge_points = edge_points_max;
		return true;
	}
	return false;
}

bool ImageDetectMethod::FilterEllipseContours_for_distance(std::vector<std::vector<cv::Point>> contours, QList<QList<float>>& ellipse_pars)
{
	std::vector<int> remove_index;
	for (int i = 0; i < ellipse_pars.size() - 1; i++)
	{
		bool need_con = false;
		for (int pp = 0; pp < remove_index.size(); pp++)
		{
			if (i == remove_index[pp])
			{
				need_con = true;
				break;
			}
		}
		if (need_con)
		{
			continue;
		}
		for (int j = i + 1; j < ellipse_pars.size(); j++)
		{
			bool need_con_c = false;
			for (int pp = 0; pp < remove_index.size(); pp++)
			{
				if (j == remove_index[pp])
				{
					need_con_c = true;
					break;
				}
			}
			if (need_con_c)
			{
				continue;
			}
			float x1 = ellipse_pars[i][0];
			float y1 = ellipse_pars[i][1];
			float x2 = ellipse_pars[j][0];
			float y2 = ellipse_pars[j][1];
			float length_of_2_points = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

			if (length_of_2_points < std::min(ellipse_pars[i][3], ellipse_pars[j][3]))
			{
				if (contours[ellipse_pars[i][6]].size() > contours[ellipse_pars[j][6]].size())
				{
					remove_index.push_back(j);
				}
				else
				{
					remove_index.push_back(i);
				}

				//if (ellipse_pars[i][3] > ellipse_pars[j][3])
				//{
				//	remove_index.push_back(j);
				//}
				//else
				//{
				//	remove_index.push_back(i);
				//}
			}
		}
	}
	QList<QList<float>> ellipse_pars_copy = ellipse_pars;
	ellipse_pars.clear();
	for (int ii = 0; ii < ellipse_pars_copy.size(); ii++)
	{
		bool need_con_c = false;
		for (int pp = 0; pp < remove_index.size(); pp++)
		{
			if (ii == remove_index[pp])
			{
				need_con_c = true;
				break;
			}
		}
		if (!need_con_c)
		{
			ellipse_pars.push_back(ellipse_pars_copy[ii]);
		}
	}
	return true;
}
bool ImageDetectMethod::FilterEllipseContoursForCSICalibrationPlane(const cv::Mat& image_mat,
	QList<QList<float>>& ellipse_pars_all,
	QList<QList<float>>& ellipse_pars, QList<QList<float>>& ellipse_pars_ori, float ratio_k)
{
	QList<float> gray_value_std_list;
	for (int i = 0; i < ellipse_pars.size(); i++)
	{
		if (EllipseGrayJudgeForPointCSI_is2Circle(ellipse_pars_all, ellipse_pars[i], ratio_k))
		{
			ellipse_pars_ori.append(ellipse_pars[i]);
		}
	}
	if (ellipse_pars_ori.size() < 3)
	{
		return false;
	}
	return true;
}

void ImageDetectMethod::LineEquation(cv::Point2f p1, cv::Point2f p2, float& A, float& B, float& C)
{
	float x1 = p1.x;
	float y1 = p1.y;
	float x2 = p2.x;
	float y2 = p2.y;

	if (x2 == x1)
	{
		A = 1;
		B = 0;
		C = -x1;

		return;
	}
	else
	{
		float k = (y2 - y1) / (x2 - x1);
		float b = y1 - k * x1;

		A = k;
		B = -1;
		C = b;
		return;
	}
}

float ImageDetectMethod::PointTpLineDistance(cv::Point2f p, float A, float B, float C)
{
	float d = abs(A * p.x + B * p.y + C) / sqrt(A * A + B * B);
	return d;
}

float ImageDetectMethod::PointToPointDistance(cv::Point2f p1, cv::Point2f p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

void ImageDetectMethod::get_bicvar(Eigen::MatrixXf value_mat, Image2D_InterpCoef* bicvar_mat)
{
	int data_size_rows = value_mat.rows();
	int data_size_cols = value_mat.cols();
	int data_size_rows_2 = value_mat.rows() / 2 + 1;
	int data_size_cols_2 = value_mat.cols() / 2 + 1;
	float* fft_data = new float[data_size_rows * data_size_cols];

	memcpy(fft_data, value_mat.data(), sizeof(float) * value_mat.size());
	fftwf_complex* fft_out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * data_size_rows * data_size_cols_2);
	fftwf_plan planner_cols = fftwf_plan_many_dft_r2c(1, &data_size_cols, data_size_rows, fft_data, NULL,
		data_size_rows, 1, fft_out, NULL, data_size_rows, 1, FFTW_ESTIMATE);
	fftwf_execute_dft_r2c(planner_cols, fft_data, fft_out);
	fftwf_destroy_plan(planner_cols);

	float* fft_data_base_cols = fftwf_alloc_real(data_size_cols);
	for (unsigned int i = 0; i < data_size_cols; i++)
	{
		fft_data_base_cols[i] = 0;
	}
	fft_data_base_cols[0] = (11.0 / 20.0);
	fft_data_base_cols[1] = (13.0 / 60.0);
	fft_data_base_cols[2] = (1.0 / 120.0);
	fft_data_base_cols[data_size_cols - 2] = 1.0 / 120.0;
	fft_data_base_cols[data_size_cols - 1] = 13.0 / 60.0;
	fftwf_complex* fft_out_base_cols = fftwf_alloc_complex(data_size_cols_2);
	fftwf_plan plan_base_cols = fftwf_plan_dft_r2c_1d(data_size_cols, fft_data_base_cols,
		fft_out_base_cols, FFTW_ESTIMATE);

	fftwf_execute(plan_base_cols);
	fftwf_destroy_plan(plan_base_cols);
	fftwf_free(fft_data_base_cols);

	float* base_cols_value = new float[data_size_cols_2];
	for (unsigned int j = 0; j < data_size_cols_2; ++j)
	{
		base_cols_value[j] = sqrt(fft_out_base_cols[j][0] * fft_out_base_cols[j][0] + fft_out_base_cols[j][1] * fft_out_base_cols[j][1]);
	}
	fftwf_free(fft_out_base_cols);
	for (unsigned int i = 0; i < data_size_rows; ++i)
	{
		for (unsigned int j = 0; j < data_size_cols_2; ++j)
		{
			fft_out[i + j * data_size_rows][0] =
				fft_out[i + j * data_size_rows][0] / base_cols_value[j]
				/ (float)data_size_cols;
			fft_out[i + j * data_size_rows][1] =
				fft_out[i + j * data_size_rows][1] / base_cols_value[j]
				/ (float)data_size_cols;
		}
	}
	fftwf_plan planner_cols_inv = fftwf_plan_many_dft_c2r(1, &data_size_cols, data_size_rows, fft_out, NULL,
		data_size_rows, 1, fft_data, NULL, data_size_rows, 1, FFTW_ESTIMATE);
	fftwf_execute(planner_cols_inv);
	fftwf_destroy_plan(planner_cols_inv);
	delete[] base_cols_value;
	fftwf_free(fft_out);
	fft_out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * data_size_rows_2 * data_size_cols);
	fftwf_plan planner_rows = fftwf_plan_many_dft_r2c(1, &data_size_rows, data_size_cols, fft_data, NULL,
		1, data_size_rows, fft_out, NULL, 1, data_size_rows_2, FFTW_ESTIMATE);
	fftwf_execute_dft_r2c(planner_rows, fft_data, fft_out);
	fftwf_destroy_plan(planner_rows);

	float* fft_data_base_rows = fftwf_alloc_real(data_size_rows);
	for (unsigned int i = 0; i < data_size_rows; i++)
	{
		fft_data_base_rows[i] = 0;
	}
	fft_data_base_rows[0] = (11.0 / 20.0);
	fft_data_base_rows[1] = (13.0 / 60.0);
	fft_data_base_rows[2] = (1.0 / 120.0);
	fft_data_base_rows[data_size_rows - 2] = 1.0 / 120.0;
	fft_data_base_rows[data_size_rows - 1] = 13.0 / 60.0;
	fftwf_complex* fft_out_base_rows = fftwf_alloc_complex(data_size_rows_2);
	fftwf_plan plan_base_rows = fftwf_plan_dft_r2c_1d(data_size_rows, fft_data_base_rows,
		fft_out_base_rows, FFTW_ESTIMATE);

	fftwf_execute(plan_base_rows);
	fftwf_destroy_plan(plan_base_rows);
	fftwf_free(fft_data_base_rows);

	float* base_rows_value = new float[data_size_rows_2];
	for (unsigned int j = 0; j < data_size_rows_2; ++j)
	{
		base_rows_value[j] = sqrt(fft_out_base_rows[j][0] * fft_out_base_rows[j][0] + fft_out_base_rows[j][1] * fft_out_base_rows[j][1]);
	}
	fftwf_free(fft_out_base_rows);
	for (unsigned int i = 0; i < data_size_cols; ++i)
	{
		for (unsigned int j = 0; j < data_size_rows_2; ++j)
		{
			fft_out[i * data_size_rows_2 + j][0] =
				fft_out[i * data_size_rows_2 + j][0] / base_rows_value[j]
				/ (float)data_size_rows;
			fft_out[i * data_size_rows_2 + j][1] =
				fft_out[i * data_size_rows_2 + j][1] / base_rows_value[j]
				/ (float)data_size_rows;
		}
	}
	fftwf_plan planner_rows_inv = fftwf_plan_many_dft_c2r(1, &data_size_rows, data_size_cols, fft_out, NULL, 1,
		data_size_rows_2, fft_data, NULL, 1, data_size_rows, FFTW_ESTIMATE);
	fftwf_execute(planner_rows_inv);
	fftwf_destroy_plan(planner_rows_inv);
	delete[] base_rows_value;
	fftwf_free(fft_out);
	for (int i = 0; i < data_size_rows; ++i)
	{
		for (int j = 0; j < data_size_cols; ++j)
		{
			bicvar_mat->eg_mat[i][j].valid = false;
		}
	}
	for (int i = 2; i < (data_size_rows - 3); ++i)
	{
		for (int j = 2; j < (data_size_cols - 3); ++j)
		{
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 0) = 0.00006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.004583333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.1191666666666667 * fft_data[(i - 1) + (j)*data_size_rows] + 0.3025 * fft_data[(i)+(j)*data_size_rows] + 0.1191666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.004583333333333333 * fft_data[(i + 2) + (j)*data_size_rows] + 0.001805555555555556 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.04694444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.004583333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.1191666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.04694444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.00006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.004583333333333333 * fft_data[(i)+(j + 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.001805555555555556 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.04694444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.1191666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.04694444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 0) = 0.009027777777777778 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.02291666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.2291666666666667 * fft_data[(i - 1) + (j)*data_size_rows] + 0.2291666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.02291666666666667 * fft_data[(i + 2) + (j)*data_size_rows] - 0.009027777777777778 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.09027777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.09027777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.009027777777777778 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.09027777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.09027777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 0) = 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.04583333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.09166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.275 * fft_data[(i)+(j)*data_size_rows] + 0.09166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.04583333333333333 * fft_data[(i + 2) + (j)*data_size_rows] + 0.01805555555555556 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.004166666666666667 * fft_data[(i)+(j - 2) * data_size_rows] - 0.1083333333333333 * fft_data[(i)+(j + 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.004166666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.1083333333333333 * fft_data[(i)+(j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 0) = 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.04583333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.09166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.09166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.04583333333333333 * fft_data[(i + 2) + (j)*data_size_rows] - 0.01805555555555556 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.01805555555555556 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 0) = 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.02291666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.09166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] + 0.1375 * fft_data[(i)+(j)*data_size_rows] - 0.09166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.02291666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.009027777777777778 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.002083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.05416666666666667 * fft_data[(i)+(j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.002083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.05416666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 0) = 0.0003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.00006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.009027777777777778 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 3) + (j - 1) * data_size_rows] - 0.004583333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.02291666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.04583333333333333 * fft_data[(i)+(j)*data_size_rows] + 0.04583333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.02291666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.004583333333333333 * fft_data[(i + 3) + (j)*data_size_rows] - 0.001805555555555556 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.0006944444444444444 * fft_data[(i)+(j - 2) * data_size_rows] - 0.01805555555555556 * fft_data[(i)+(j + 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.009027777777777778 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 3) + (j + 1) * data_size_rows] - 0.00006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i)+(j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 3) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 3) + (j - 2) * data_size_rows] - 0.001805555555555556 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.01805555555555556 * fft_data[(i)+(j - 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 1) = 0.003472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.009027777777777778 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.09027777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.02291666666666667 * fft_data[(i)+(j - 2) * data_size_rows] + 0.2291666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.09027777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.02291666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.009027777777777778 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.09027777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.2291666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.09027777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 1) = 0.001736111111111111 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.1736111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.1736111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.1736111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.1736111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 1) = 0.03472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] - 0.2083333333333333 * fft_data[(i)+(j + 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.2083333333333333 * fft_data[(i)+(j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 1) = 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.03472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 1) = 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.01736111111111111 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.01041666666666667 * fft_data[(i)+(j - 2) * data_size_rows] + 0.1041666666666667 * fft_data[(i)+(j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.01041666666666667 * fft_data[(i)+(j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.1041666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 1) = 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 3) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.01736111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i)+(j - 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i)+(j + 1) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.01736111111111111 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 3) + (j + 1) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i)+(j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 3) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 3) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.03472222222222222 * fft_data[(i)+(j - 1) * data_size_rows] - 0.03472222222222222 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 2) = 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.004166666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.1083333333333333 * fft_data[(i - 1) + (j)*data_size_rows] - 0.275 * fft_data[(i)+(j)*data_size_rows] - 0.1083333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.004166666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.04583333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.09166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.04583333333333333 * fft_data[(i)+(j + 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.09166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 2) = 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.02083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.2083333333333333 * fft_data[(i - 1) + (j)*data_size_rows] - 0.2083333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.02083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 2) = 0.006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.04166666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.08333333333333333 * fft_data[(i - 1) + (j)*data_size_rows] + 0.25 * fft_data[(i)+(j)*data_size_rows] - 0.08333333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.04166666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j - 2) * data_size_rows] - 0.08333333333333333 * fft_data[(i)+(j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.08333333333333333 * fft_data[(i)+(j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 2) = 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.04166666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.08333333333333333 * fft_data[(i - 1) + (j)*data_size_rows] + 0.08333333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.04166666666666667 * fft_data[(i + 2) + (j)*data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 2) = 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.02083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.08333333333333333 * fft_data[(i - 1) + (j)*data_size_rows] - 0.125 * fft_data[(i)+(j)*data_size_rows] + 0.08333333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.02083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 2) = 0.003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 3) + (j - 1) * data_size_rows] + 0.004166666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.02083333333333333 * fft_data[(i - 1) + (j)*data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j)*data_size_rows] - 0.04166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.02083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] - 0.004166666666666667 * fft_data[(i + 3) + (j)*data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i)+(j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i)+(j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 3) + (j + 1) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i)+(j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 3) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 3) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i)+(j - 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 3) = 0.001388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.01805555555555556 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.04583333333333333 * fft_data[(i)+(j - 2) * data_size_rows] - 0.09166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.04583333333333333 * fft_data[(i)+(j + 2) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.01805555555555556 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.09166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 3) = 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.03472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.03472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 3) = 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j - 2) * data_size_rows] + 0.08333333333333333 * fft_data[(i)+(j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.08333333333333333 * fft_data[(i)+(j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 3) = 0.006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 3) = 0.01388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 3) = 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 3) + (j - 1) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i)+(j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i)+(j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 3) + (j + 1) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i)+(j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 3) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i + 3) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i)+(j - 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 4) = 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.002083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.05416666666666667 * fft_data[(i - 1) + (j)*data_size_rows] + 0.1375 * fft_data[(i)+(j)*data_size_rows] + 0.05416666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.002083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.02291666666666667 * fft_data[(i)+(j - 2) * data_size_rows] - 0.09166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.02291666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.09166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.03611111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 4) = 0.1041666666666667 * fft_data[(i + 1) + (j)*data_size_rows] - 0.01736111111111111 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.01041666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.1041666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.01041666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.06944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.06944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 4) = 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.02083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.04166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.125 * fft_data[(i)+(j)*data_size_rows] + 0.04166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.02083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.08333333333333333 * fft_data[(i)+(j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.08333333333333333 * fft_data[(i)+(j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 4) = 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.02083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.04166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.04166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.02083333333333333 * fft_data[(i + 2) + (j)*data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 4) = 0.001736111111111111 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.01041666666666667 * fft_data[(i - 2) + (j)*data_size_rows] - 0.04166666666666667 * fft_data[(i - 1) + (j)*data_size_rows] + 0.0625 * fft_data[(i)+(j)*data_size_rows] - 0.04166666666666667 * fft_data[(i + 1) + (j)*data_size_rows] + 0.01041666666666667 * fft_data[(i + 2) + (j)*data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.01041666666666667 * fft_data[(i)+(j - 2) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.01041666666666667 * fft_data[(i)+(j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.02777777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 4) = 0.001736111111111111 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 3) + (j - 1) * data_size_rows] - 0.002083333333333333 * fft_data[(i - 2) + (j)*data_size_rows] + 0.01041666666666667 * fft_data[(i - 1) + (j)*data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j)*data_size_rows] + 0.02083333333333333 * fft_data[(i + 1) + (j)*data_size_rows] - 0.01041666666666667 * fft_data[(i + 2) + (j)*data_size_rows] + 0.002083333333333333 * fft_data[(i + 3) + (j)*data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i)+(j - 2) * data_size_rows] + 0.01388888888888889 * fft_data[(i)+(j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 3) + (j + 1) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i)+(j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 3) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 3) + (j - 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i)+(j - 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(0, 5) = 0.0003472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.001805555555555556 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.00006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j)*data_size_rows] - 0.01805555555555556 * fft_data[(i - 1) + (j)*data_size_rows] - 0.04583333333333333 * fft_data[(i)+(j)*data_size_rows] - 0.01805555555555556 * fft_data[(i + 1) + (j)*data_size_rows] - 0.0006944444444444444 * fft_data[(i + 2) + (j)*data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.004583333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.04583333333333333 * fft_data[(i)+(j + 1) * data_size_rows] + 0.01805555555555556 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.009027777777777778 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.02291666666666667 * fft_data[(i)+(j + 2) * data_size_rows] - 0.009027777777777778 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.001805555555555556 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.00006944444444444444 * fft_data[(i - 2) + (j + 3) * data_size_rows] + 0.001805555555555556 * fft_data[(i - 1) + (j + 3) * data_size_rows] + 0.004583333333333333 * fft_data[(i)+(j + 3) * data_size_rows] + 0.001805555555555556 * fft_data[(i + 1) + (j + 3) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 2) + (j + 3) * data_size_rows] - 0.00006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.02291666666666667 * fft_data[(i)+(j - 1) * data_size_rows] + 0.009027777777777778 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(1, 5) = 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j)*data_size_rows] + 0.03472222222222222 * fft_data[(i - 1) + (j)*data_size_rows] - 0.03472222222222222 * fft_data[(i + 1) + (j)*data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j)*data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.03472222222222222 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.03472222222222222 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.01736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.01736111111111111 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j + 3) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 1) + (j + 3) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j + 3) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 3) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.01736111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.01736111111111111 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(2, 5) = 0.003472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j)*data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j)*data_size_rows] + 0.04166666666666667 * fft_data[(i)+(j)*data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j)*data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j)*data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.004166666666666667 * fft_data[(i)+(j - 2) * data_size_rows] - 0.04166666666666667 * fft_data[(i)+(j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j + 3) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 1) + (j + 3) * data_size_rows] - 0.004166666666666667 * fft_data[(i)+(j + 3) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 1) + (j + 3) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 3) * data_size_rows] - 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(3, 5) = 0.0006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 2) + (j)*data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j)*data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j)*data_size_rows] - 0.006944444444444444 * fft_data[(i + 2) + (j)*data_size_rows] - 0.006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j + 3) * data_size_rows] + 0.001388888888888889 * fft_data[(i - 1) + (j + 3) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j + 3) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 2) + (j + 3) * data_size_rows] - 0.0006944444444444444 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(4, 5) = 0.001388888888888889 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i - 2) + (j)*data_size_rows] + 0.01388888888888889 * fft_data[(i - 1) + (j)*data_size_rows] - 0.02083333333333333 * fft_data[(i)+(j)*data_size_rows] + 0.01388888888888889 * fft_data[(i + 1) + (j)*data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j)*data_size_rows] + 0.003472222222222222 * fft_data[(i - 2) + (j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i - 1) + (j + 1) * data_size_rows] - 0.002083333333333333 * fft_data[(i)+(j - 2) * data_size_rows] + 0.02083333333333333 * fft_data[(i)+(j + 1) * data_size_rows] - 0.01388888888888889 * fft_data[(i + 1) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 2) + (j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i - 1) + (j + 2) * data_size_rows] - 0.01041666666666667 * fft_data[(i)+(j + 2) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] + 0.001388888888888889 * fft_data[(i + 1) + (j - 2) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j + 3) * data_size_rows] - 0.001388888888888889 * fft_data[(i - 1) + (j + 3) * data_size_rows] + 0.002083333333333333 * fft_data[(i)+(j + 3) * data_size_rows] - 0.001388888888888889 * fft_data[(i + 1) + (j + 3) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j + 3) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 2) + (j - 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i - 1) + (j - 1) * data_size_rows] + 0.01041666666666667 * fft_data[(i)+(j - 1) * data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j - 1) * data_size_rows];
			bicvar_mat->eg_mat[i][j].InterpCoef_mat(5, 5) = 0.00006944444444444444 * fft_data[(i - 2) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 1) + (j - 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i + 2) + (j - 1) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 3) + (j - 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i - 2) + (j)*data_size_rows] - 0.003472222222222222 * fft_data[(i - 1) + (j)*data_size_rows] + 0.006944444444444444 * fft_data[(i)+(j)*data_size_rows] - 0.006944444444444444 * fft_data[(i + 1) + (j)*data_size_rows] + 0.003472222222222222 * fft_data[(i + 2) + (j)*data_size_rows] - 0.0006944444444444444 * fft_data[(i + 3) + (j)*data_size_rows] - 0.0006944444444444444 * fft_data[(i - 2) + (j + 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i - 1) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i)+(j - 2) * data_size_rows] - 0.006944444444444444 * fft_data[(i)+(j + 1) * data_size_rows] + 0.006944444444444444 * fft_data[(i + 1) + (j + 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 2) + (j + 1) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 3) + (j + 1) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 2) + (j + 2) * data_size_rows] - 0.001736111111111111 * fft_data[(i - 1) + (j + 2) * data_size_rows] + 0.003472222222222222 * fft_data[(i)+(j + 2) * data_size_rows] - 0.003472222222222222 * fft_data[(i + 1) + (j + 2) * data_size_rows] + 0.001736111111111111 * fft_data[(i + 2) + (j + 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 3) + (j + 2) * data_size_rows] - 0.0006944444444444444 * fft_data[(i + 1) + (j - 2) * data_size_rows] - 0.00006944444444444444 * fft_data[(i - 2) + (j + 3) * data_size_rows] + 0.0003472222222222222 * fft_data[(i - 1) + (j + 3) * data_size_rows] - 0.0006944444444444444 * fft_data[(i)+(j + 3) * data_size_rows] + 0.0006944444444444444 * fft_data[(i + 1) + (j + 3) * data_size_rows] - 0.0003472222222222222 * fft_data[(i + 2) + (j + 3) * data_size_rows] + 0.00006944444444444444 * fft_data[(i + 3) + (j + 3) * data_size_rows] + 0.0003472222222222222 * fft_data[(i + 2) + (j - 2) * data_size_rows] - 0.00006944444444444444 * fft_data[(i + 3) + (j - 2) * data_size_rows] - 0.0003472222222222222 * fft_data[(i - 2) + (j - 1) * data_size_rows] + 0.001736111111111111 * fft_data[(i - 1) + (j - 1) * data_size_rows] - 0.003472222222222222 * fft_data[(i)+(j - 1) * data_size_rows] + 0.003472222222222222 * fft_data[(i + 1) + (j - 1) * data_size_rows];

			bicvar_mat->eg_mat[i][j].valid = true;
		}
	}

	delete[] fft_data;
	
}
void ImageDetectMethod::get_hess_pixel(Eigen::MatrixXf value_mat, int pixel_x, int pixel_y, Image2D_InterpCoef* bic_var, 
	int window_R, Eigen::MatrixXf Cross_matrix, int Num_weight, Hess_struct* hess)
{
	int ii = pixel_x;
	int jj = pixel_y;

	float mean_value = 0;

	for (int pp = ii - window_R; pp < ii + window_R + 1; ++pp)
	{
		for (int qq = jj - window_R; qq < jj + window_R + 1; ++qq)
		{
			if (Cross_matrix(pp + window_R - ii, qq + window_R - jj))
			{
				mean_value += (value_mat)(pp, qq);
			}
		}
	}
	mean_value /= Num_weight;
	hess->mean_val = mean_value;
	float dev_val = 0;
	for (int pp = ii - window_R; pp < ii + window_R + 1; ++pp)
	{
		for (int qq = jj - window_R; qq < jj + window_R + 1; ++qq)
		{
			if (Cross_matrix(pp + window_R - ii, qq + window_R - jj))
			{
				dev_val += pow((value_mat)(pp, qq) - mean_value, 2);
			}

		}
	}
	if (dev_val == 0)
	{
		hess->dev_val = NAN;
		return;
	}
	dev_val = 1.0 / sqrt(dev_val);
	hess->dev_val = dev_val;

	float delta_x, delta_y;
	hess->Hess_mat = Matrix6f::Zero(6, 6);
	float dv, du, dvp1, dup1, dvp2, dup2;

	hess->Dif_inf.Dv = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	hess->Dif_inf.Du = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	hess->Dif_inf.Dvp1 = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	hess->Dif_inf.Dvp2 = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	hess->Dif_inf.Dup1 = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	hess->Dif_inf.Dup2 = Eigen::MatrixXf::Zero(2 * window_R + 1, 2 * window_R + 1);
	for (int pp = ii - window_R; pp < ii + window_R + 1; ++pp)
	{
		for (int qq = jj - window_R; qq < jj + window_R + 1; ++qq)
		{
			if (Cross_matrix(pp + window_R - ii, qq + window_R - jj))
			{
				int index_x = pp - ii + window_R;
				int index_y = qq - jj + window_R;
				delta_x = pp - ii;
				delta_y = qq - jj;
				dv = bic_var->eg_mat[pp][qq].InterpCoef_mat(1, 0);
				du = bic_var->eg_mat[pp][qq].InterpCoef_mat(0, 1);
				dvp1 = dv * delta_x;
				dvp2 = dv * delta_y;
				dup1 = du * delta_x;
				dup2 = du * delta_y;
				hess->Dif_inf.Dv(index_x, index_y) = dv;
				hess->Dif_inf.Du(index_x, index_y) = du;
				hess->Dif_inf.Dvp1(index_x, index_y) = dvp1;
				hess->Dif_inf.Dup1(index_x, index_y) = dup1;
				hess->Dif_inf.Dvp2(index_x, index_y) = dvp2;
				hess->Dif_inf.Dup2(index_x, index_y) = dup2;
				hess->Hess_mat(0, 0) += dv * dv;
				hess->Hess_mat(1, 0) += dv * du;
				hess->Hess_mat(2, 0) += dv * dvp1;
				hess->Hess_mat(3, 0) += dv * dvp2;
				hess->Hess_mat(4, 0) += dv * dup1;
				hess->Hess_mat(5, 0) += dv * dup2;

				hess->Hess_mat(1, 1) += du * du;
				hess->Hess_mat(2, 1) += du * dvp1;
				hess->Hess_mat(3, 1) += du * dvp2;
				hess->Hess_mat(4, 1) += du * dup1;
				hess->Hess_mat(5, 1) += du * dup2;

				hess->Hess_mat(2, 2) += dvp1 * dvp1;
				hess->Hess_mat(3, 2) += dvp1 * dvp2;
				hess->Hess_mat(4, 2) += dvp1 * dup1;
				hess->Hess_mat(5, 2) += dvp1 * dup2;

				hess->Hess_mat(3, 3) += dvp2 * dvp2;
				hess->Hess_mat(4, 3) += dvp2 * dup1;
				hess->Hess_mat(5, 3) += dvp2 * dup2;

				hess->Hess_mat(4, 4) += dup1 * dup1;
				hess->Hess_mat(5, 4) += dup1 * dup2;

				hess->Hess_mat(5, 5) += dup2 * dup2;
			}
		}
	}
	cv::Mat mat_23f;
	Eigen::MatrixXf Img_reff = hess->Dif_inf.Dv;
	cv::eigen2cv(Img_reff, mat_23f);

	for (int pp = 0; pp < 6; ++pp)
	{
		for (int qq = 0; qq <= pp; ++qq)
		{
			hess->Hess_mat(pp, qq) = hess->Hess_mat(pp, qq) * 2.0 * dev_val * dev_val;
		}
	}
	for (int pp = 0; pp < 6; ++pp)
	{
		for (int qq = 0; qq < pp; ++qq)
		{
			hess->Hess_mat(qq, pp) = hess->Hess_mat(pp, qq);
		}
	}
	hess->valid = true;
}

void ImageDetectMethod::decode_speckle(cv::Mat part, Image2D_InterpCoef B_cof, int left_b, int up_b
	, Code_speckle& Code_inf, float ratio_k, float min_cc, int code_bites)
{
	Code_inf.code_value = -1;
	int sample_v = 10;
	float dis_interval = Code_inf.sepckle_pos.R / ratio_k * (1 - ratio_k) / ((float)sample_v + 1.0);
	float angle_interval = 360.0 / (float)code_bites / ((float)sample_v + 1.0);
	std::vector<std::vector<float>> gray_list;
	std::vector<std::vector<float>> gray_list_x;
	std::vector<std::vector<float>> gray_list_y;
	std::vector<float> gray_mean;
	std::vector<float> gray_stdinv;
	float F[6];
	float V[6];

	for (int ii = 0; ii < code_bites; ii++)
	{
		float mean_val = 0;
		float dev_val = 0;
		float mean_val_num = 0;
		std::vector<float> gray_list_now;
		//std::vector<float> gray_listx_now;
		//std::vector<float> gray_listy_now;
		for (int angle_p = 0; angle_p < sample_v; angle_p++)
		{
			for (int dis_p = 0; dis_p < sample_v; dis_p++)
			{
				float R_now = Code_inf.sepckle_pos.R + dis_interval * (dis_p + 1);
				//0.017453292519943=2pi/360;
				float theta_now = (((float)ii) * 360.0 / (float)code_bites + angle_interval * (angle_p + 1)) * 0.017453292519943;
				float y_now = -R_now * cos(theta_now);
				float x_now = -R_now * sin(theta_now);
				float tran_x = -up_b + Code_inf.sepckle_pos.x + Code_inf.sepckle_pos.dx + (Code_inf.sepckle_pos.dxp1 + 1) * x_now + Code_inf.sepckle_pos.dxp2 * y_now;
				float tran_y = -left_b + Code_inf.sepckle_pos.y + Code_inf.sepckle_pos.dy + Code_inf.sepckle_pos.dyp1 * x_now + (Code_inf.sepckle_pos.dyp2 + 1) * y_now;
				//gray_listx_now.push_back(tran_x+ up_b);
				//gray_listy_now.push_back(tran_y+ left_b);
				int tran_x_floor = floor(tran_x);
				int tran_y_floor = floor(tran_y);
				float tran_x_delta = tran_x - tran_x_floor;
				float tran_y_delta = tran_y - tran_y_floor;
				if (tran_x_floor < 0 || tran_x_floor >= B_cof.height || tran_y_floor < 0 || tran_y_floor >= B_cof.width)
				{
					gray_list_now.push_back(NAN);
					continue;
				}
				F[1] = tran_x_delta;
				F[2] = F[1] * tran_x_delta;
				F[3] = F[2] * tran_x_delta;
				F[4] = F[3] * tran_x_delta;
				F[5] = F[4] * tran_x_delta;
				V[1] = tran_y_delta;
				V[2] = V[1] * tran_y_delta;
				V[3] = V[2] * tran_y_delta;
				V[4] = V[3] * tran_y_delta;
				V[5] = V[4] * tran_y_delta;
				float v_NOW =
					(B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 5) * V[5])
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 5) * V[5]) * F[1]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 5) * V[5]) * F[2]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 5) * V[5]) * F[3]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 5) * V[5]) * F[4]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 5) * V[5]) * F[5];
				mean_val+= v_NOW;
				gray_list_now.push_back(v_NOW);
				mean_val_num += 1.0;
			}
		}
		mean_val /= mean_val_num;
		for (int ss = 0; ss < gray_list_now.size(); ss++)
		{
			if (!isnan(gray_list_now[ss]))
			{
				dev_val += pow((gray_list_now[ss]) - mean_val, 2);
			}
		}
		dev_val = sqrt(dev_val);
		if (dev_val > 0)
		{
			dev_val = 1 / dev_val;
			gray_mean.push_back(mean_val);
			gray_stdinv.push_back(dev_val);
		}
		else
		{
			gray_mean.push_back(NAN);
			gray_stdinv.push_back(NAN);
		}
		//gray_list_x.push_back(gray_listx_now);
		//gray_list_y.push_back(gray_listy_now);
		gray_list.push_back(gray_list_now);
	}
	std::vector<float> CC_list;
	bool is_canbe_code=true;
	//std::ofstream outFile("C:\\example.txt");
	//if (!outFile) {
	//	std::cerr << "Failed to open the file." << std::endl;
	//	return; 
	//}
	//for (int jj=0; jj < gray_list[0].size(); jj++)
	//{
	//	for (int ii = 0; ii < code_bites; ii++)
	//	{
	//		outFile << gray_list_x[ii][jj] << "\t" << gray_list_y[ii][jj] << "\t" << gray_list[ii][jj] << "\t";
	//	}
	//	outFile << "\n";
	//}
	//outFile.close();
	for (int ii = 1; ii < code_bites; ii++)
	{
		float norm_dif = cal_cc(gray_list[ii], gray_mean[ii], gray_stdinv[ii], gray_list[0], gray_mean[0], gray_stdinv[0]);
		CC_list.push_back(norm_dif);
		if (abs(norm_dif) < min_cc)
		{
			is_canbe_code = false;
			break;
		}
	}
	if (is_canbe_code)
	{
		int decimalValue = 0;
		for (float bits : CC_list) {
			bool bit = bits > 0 ? false : true;
			decimalValue = (decimalValue << 1) | bit;
		}
		Code_inf.code_value = decimalValue;
	}
	else
	{
		Code_inf.code_value = -1;
	}
}
float ImageDetectMethod::cal_cc(std::vector<float> v1, float mean_v1, float invstd_v1, std::vector<float> v2, float mean_v2, float invstd_v2)
{
	float norm_dif = 0;
	for (int ii = 0; ii < v1.size(); ii++)
	{
		if (!isnan(v1[ii]) && !isnan(v2[ii]))
		{
			norm_dif += (v1[ii] - mean_v1) * (v2[ii] - mean_v2);
		}
	}
	norm_dif *= (invstd_v1 * invstd_v2);
	return norm_dif;
}
bool ImageDetectMethod::detectCheckerboard(cv::Mat I, std::vector<cv::Point2f>& corner_points, cv::Size& wh_check, double peakThreshold, bool highDistortion, bool usePartial)
{
	if (I.channels() == 3)
	{
		cv::cvtColor(I, I, CV_BGR2GRAY);
	}
	coder::array<double, 2U> imagePoints;
	coder::array<unsigned char, 2U> b_I;
	b_I.set_size(I.rows, I.cols);
	for (int idx0{ 0 }; idx0 < b_I.size(0); idx0++) {
		for (int idx1{ 0 }; idx1 < b_I.size(1); idx1++) {
			b_I[idx0 + b_I.size(0) * idx1] = I.at<uchar>(idx0, idx1);
		}
	}
	double boardSize[2];
	boolean_T B1_tmp;
	boolean_T imagesUsed;
	get_chessborad_pixel(b_I, peakThreshold, highDistortion, usePartial, imagePoints,
		boardSize, &imagesUsed);
	if (imagesUsed)
	{
		corner_points.clear();
		for (int idx0{ 0 }; idx0 < imagePoints.size(0); idx0++)
		{
			corner_points.push_back(cv::Point2d(imagePoints[idx0], imagePoints[idx0 + imagePoints.size(0)]));
			wh_check.width = boardSize[0];
			wh_check.height = boardSize[1];
		}
		return true;
	}
	else
	{
		return false;
	}
}

bool ImageDetectMethod::detectcodecircle(cv::Mat ori_image, cv::Mat& code_point_mat, std::vector<std::vector<cv::Point2f>>& contours_pixel,
	float ratio_k, float ratio_k1, float ratio_k2,
	float min_radius, float max_radius, float ellipse_error_pixel /*=0.5*/,
	MarkPointColorType color_type /*= BlackDownWhiteUp*/, CodePointBitesType code_bites_type /*=CodeBites15*/,
	DetectContoursMethod image_process_method /*= CANNY_Method*/,
	SubPixelPosMethod subpixel_pos_method,
	double max_aspect_ratio, int min_points, int min_contour_num,
	float delta_Mt, float fore_stdDev, float back_stdDev)
{
	if (!ori_image.data)        // 判断图片调入是否成功
		return false;        // 调入图片失败则退出

	if (ori_image.channels() == 3)
	{
		cv::cvtColor(ori_image, ori_image, CV_BGR2GRAY);
	}
	contours_pixel.clear();
	QList<QList<float>> ellipse_pars_all;
	cv::Mat processed_image_mat;
	ImagePreprocess(ori_image, processed_image_mat);

	if (color_type == MarkPointColorType::BlackDownWhiteUp || color_type == MarkPointColorType::Uncertainty)
	{
		std::vector<std::vector<cv::Point>> contours;
		QList<QList<float>> ellipse_pars;
		DetectClosedContours(processed_image_mat, contours, image_process_method);
		FilterEllipseContours(contours, min_radius, max_radius,
			ellipse_error_pixel, ellipse_pars, max_aspect_ratio, min_points, min_contour_num);
		FilterEllipseContoursForCodePoint(processed_image_mat, ratio_k, ratio_k1, ratio_k2,
			ellipse_pars, delta_Mt, fore_stdDev, back_stdDev);
		int* default_id_array_ptr = NULL;
		int default_id_array_size;
		default_id_array_ptr = ReturnDefualtIdArray(default_id_array_size, code_bites_type);
		if (default_id_array_ptr == NULL)
		{
			return false;
		}

		int uncodePoint_id = 0;
		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			int code_id;
			bool is_decode = Decoding20140210(processed_image_mat, code_id, ellipse_pars[i][0], ellipse_pars[i][1], ellipse_pars[i][2], ellipse_pars[i][3], ellipse_pars[i][4],
				ratio_k1, ratio_k2, BlackDownWhiteUp, code_bites_type, delta_Mt);

			bool is_code_point = false;
			if (is_decode == true)
			{
				for (int j = 0; j < default_id_array_size; j++)
				{
					int id = *(default_id_array_ptr + j);
					if (code_id == *(default_id_array_ptr + j))
					{
						is_code_point = true;

						ellipse_pars[i].append(j);
						ellipse_pars[i].append(1);

						break;
					}
				}
			}

			if (is_code_point == false)
			{
				bool 	is_uncodepoint = true;
				if (is_uncodepoint == true)
				{
					ellipse_pars[i].append(uncodePoint_id);
					ellipse_pars[i].append(0);
					uncodePoint_id++;
				}
				else
				{
					ellipse_pars.removeAt(i);
					i--;
				}
			}
		}

		std::vector<std::vector<cv::Point2f>> subpixel_edge_contours;
		std::vector<int> subpixel_edge_contours_index;
		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			float sub_pixel_x, sub_pixel_y;
			std::vector<cv::Point2f> edge_contour;
			if (FindSubPixelPosOfCircleCenter(processed_image_mat, ellipse_pars[i][0], ellipse_pars[i][1], ellipse_pars[i][2],
				ellipse_pars[i][3], ellipse_pars[i][4], contours[ellipse_pars[i][6]], sub_pixel_x, sub_pixel_y,
				edge_contour, subpixel_pos_method))
			{
				ellipse_pars[i][0] = sub_pixel_x;
				ellipse_pars[i][1] = sub_pixel_y;
				subpixel_edge_contours.push_back(edge_contour);
			}
			else
			{
				edge_contour.clear();
				for (int tt = 0; tt < contours[ellipse_pars[i][6]].size(); tt++)
				{
					edge_contour.push_back(cv::Point2f(contours[ellipse_pars[i][6]][tt].x, contours[ellipse_pars[i][6]][tt].y));
				}
				subpixel_edge_contours.push_back(edge_contour);
			}
			subpixel_edge_contours_index.push_back(ellipse_pars[i][6]);
		}

		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			if (ellipse_pars[i][8] > 0)
			{
				ellipse_pars_all.append(ellipse_pars[i]);
				for (int j = 0; j < subpixel_edge_contours_index.size(); j++)
				{
					if (subpixel_edge_contours_index[j] == ellipse_pars[i][6])
					{
						contours_pixel.push_back(subpixel_edge_contours[j]);
					}
				}
			}
		}
	}
	if (color_type == MarkPointColorType::WhiteDownBlackUp || color_type == MarkPointColorType::Uncertainty)
	{
		std::vector<std::vector<cv::Point>> contours;
		QList<QList<float>> ellipse_pars;
		DetectClosedContours(processed_image_mat, contours, image_process_method);
		FilterEllipseContours(contours, min_radius, max_radius,
			ellipse_error_pixel, ellipse_pars, max_aspect_ratio, min_points, min_contour_num);
		FilterEllipseContoursForCodePoint(processed_image_mat, ratio_k, ratio_k1, ratio_k2,
			ellipse_pars, delta_Mt, fore_stdDev, back_stdDev);
		int* default_id_array_ptr = NULL;
		int default_id_array_size;
		default_id_array_ptr = ReturnDefualtIdArray(default_id_array_size, code_bites_type);
		if (default_id_array_ptr == NULL)
		{
			return false;
		}

		int uncodePoint_id = 0;
		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			int code_id;
			bool is_decode = Decoding20140210(processed_image_mat, code_id, ellipse_pars[i][0], ellipse_pars[i][1], ellipse_pars[i][2], ellipse_pars[i][3], ellipse_pars[i][4],
				ratio_k1, ratio_k2, WhiteDownBlackUp, code_bites_type, delta_Mt);

			bool is_code_point = false;
			if (is_decode == true)
			{
				for (int j = 0; j < default_id_array_size; j++)
				{
					int id = *(default_id_array_ptr + j);
					if (code_id == *(default_id_array_ptr + j))
					{
						is_code_point = true;

						ellipse_pars[i].append(j);
						ellipse_pars[i].append(1);

						break;
					}
				}
			}

			if (is_code_point == false)
			{
				bool 	is_uncodepoint = true;
				if (is_uncodepoint == true)
				{
					ellipse_pars[i].append(uncodePoint_id);
					ellipse_pars[i].append(0);
					uncodePoint_id++;
				}
				else
				{
					ellipse_pars.removeAt(i);
					i--;
				}
			}
		}

		std::vector<std::vector<cv::Point2f>> subpixel_edge_contours;
		std::vector<int> subpixel_edge_contours_index;
		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			float sub_pixel_x, sub_pixel_y;
			std::vector<cv::Point2f> edge_contour;
			if (FindSubPixelPosOfCircleCenter(processed_image_mat, ellipse_pars[i][0], ellipse_pars[i][1], ellipse_pars[i][2],
				ellipse_pars[i][3], ellipse_pars[i][4], contours[ellipse_pars[i][6]], sub_pixel_x, sub_pixel_y,
				edge_contour, subpixel_pos_method))
			{
				ellipse_pars[i][0] = sub_pixel_x;
				ellipse_pars[i][1] = sub_pixel_y;
				subpixel_edge_contours.push_back(edge_contour);
			}
			else
			{
				edge_contour.clear();
				for (int tt = 0; tt < contours[ellipse_pars[i][6]].size(); tt++)
				{
					edge_contour.push_back(cv::Point2f(contours[ellipse_pars[i][6]][tt].x, contours[ellipse_pars[i][6]][tt].y));
				}
				subpixel_edge_contours.push_back(edge_contour);
			}
			subpixel_edge_contours_index.push_back(ellipse_pars[i][6]);
		}

		for (int i = 0; i < ellipse_pars.size(); i++)
		{
			if (ellipse_pars[i][8] > 0)
			{
				ellipse_pars_all.append(ellipse_pars[i]);
				for (int j = 0; j < subpixel_edge_contours_index.size(); j++)
				{
					if (subpixel_edge_contours_index[j] == ellipse_pars[i][6])
					{
						contours_pixel.push_back(subpixel_edge_contours[j]);
					}
				}
			}
		}
	}
//ellipse_pars - n*6  center_x,center_y,r_a,r_b,angle_inPI,ellipse_error,contours_index,ID,code_type(0- uncode,1- code)
	code_point_mat = cv::Mat();
	for (int i = 0; i < ellipse_pars_all.size(); i++)
	{
		float a[7] = { ellipse_pars_all[i][7],ellipse_pars_all[i][0],ellipse_pars_all[i][1],ellipse_pars_all[i][5],
			ellipse_pars_all[i][2],ellipse_pars_all[i][3],ellipse_pars_all[i][4] };
		cv::Mat mat = cv::Mat(1, 7, CV_32F, a);
		code_point_mat.push_back(mat);
	}
	return true;
}
//
bool ImageDetectMethod::detectcodespeckle(cv::Mat ori_image, cv::Mat image_key, cv::Mat image_key_mask1, cv::Mat image_key_mask2,
	std::vector<Code_speckle> &Code_inf, float ratio_k, float min_cc_key_inital, float min_cc_key, float min_cc_code, int code_bites,
	int temple_R, int angle_search,
	float min_radius, float max_radius, float ellipse_error_pixel,
	DetectContoursMethod image_process_method
	, double max_aspect_ratio, int min_points, int min_contour_num)
{

	if (!ori_image.data)        // 判断图片调入是否成功
		return false;        // 调入图片失败则退出

	if (ori_image.channels() == 3)
	{
		cv::cvtColor(ori_image, ori_image, CV_BGR2GRAY);
	}

	int re_scale_r = temple_R;
	image_key.convertTo(image_key, CV_64F);
	double scale_cof = (double)re_scale_r / ((image_key_mask1.rows - 1) / 2.0);
	//GaussianBlur(image_key, image_key, cv::Size(51, 51), 10, 10);
	image_key = Image_Resize(image_key, ceil((image_key.rows - 1) / 2.0 * scale_cof) * 2 + 1
		, ceil((image_key.rows - 1) / 2.0 * scale_cof) * 2 + 1
		, 2, true);
	//cv::resize(image_key, image_key, cv::Size(ceil((image_key.rows - 1) / 2.0 * scale_cof) * 2 + 1
	//	, ceil((image_key.rows - 1) / 2.0 * scale_cof) * 2 + 1), cv::INTER_LANCZOS4);
	std::vector<std::vector<cv::Point>> contours_pixel;	
	cv::Mat code_point_mat;
	detectcircle_rough(ori_image, code_point_mat, contours_pixel,
		min_radius, max_radius, ellipse_error_pixel,
		image_process_method, max_aspect_ratio, min_points, min_contour_num);
	
	Eigen::MatrixXf Img_ref;
	cv::cv2eigen(image_key, Img_ref);
	int window_R = re_scale_r;
	int re_scale_for_circle = scale_cof * ((image_key_mask2.rows - 1) / 2.0);
	int R_S = (image_key.rows - 1) / 2;

	Eigen::MatrixXf Cross_matrix = Eigen::MatrixXf::Zero(window_R * 2 + 1, window_R * 2 + 1);
	int duak_R = window_R * window_R;
	int Num_weight = 0;
	for (int pp = -window_R; pp < window_R + 1; ++pp)
	{
		for (int qq = -window_R; qq < window_R + 1; ++qq)
		{
			if ((pp * pp + qq * qq) <= duak_R)
			{
				Cross_matrix(pp + window_R, qq + window_R) = 1;
				Num_weight++;
			}
			else
			{
				Cross_matrix(pp + window_R, qq + window_R) = 0;
			}
		}
	}
	Image2D_InterpCoef bic_var_ref(image_key.cols, image_key.rows);
	Hess_struct hess;
	get_bicvar(Img_ref, &bic_var_ref);

	//计算Hess
	get_hess_pixel(Img_ref, R_S, R_S, &bic_var_ref
		, window_R, Cross_matrix, Num_weight, &hess);
	//std::cout << hess.Hess_mat << std::endl;
	int num = 0;
	std::vector<Code_speckle> Code_inf_temp;
	for (int num = 0; num < contours_pixel.size(); num++)
	{
		bool repeat_area = false;
		for (int pp = 0; pp < Code_inf_temp.size(); pp++)
		{
			float if_delta_x = ((code_point_mat.at<float>(num, 2) - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * Code_inf_temp[pp].sepckle_pos.dyp1
				- (code_point_mat.at<float>(num, 1) - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp1 + 1))
				/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
			float if_delta_y = -((code_point_mat.at<float>(num, 2) - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1)
				- (code_point_mat.at<float>(num, 1) - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp2))
				/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
			if (if_delta_x * if_delta_x + if_delta_y * if_delta_y < Code_inf_temp[pp].sepckle_pos.R * Code_inf_temp[pp].sepckle_pos.R)
			{
				repeat_area = true;
				break;
			}
		}
		if (repeat_area)
		{
			continue;
		}
		Pos_ellispe pos_circle_now(code_point_mat.at<float>(num, 1), code_point_mat.at<float>(num, 2)
			, code_point_mat.at<float>(num, 4), code_point_mat.at<float>(num, 5)
			, code_point_mat.at<float>(num, 6), code_point_mat.at<float>(num, 3));
		float max_axis = code_point_mat.at<float>(num, 4) > code_point_mat.at<float>(num, 5) ? code_point_mat.at<float>(num, 4) : code_point_mat.at<float>(num, 5);
		max_axis = max_axis < 5 ? 5 : max_axis;
		//max_axis *= (1.5 / ratio_k);
		max_axis *= (1.5 * (float)R_S/(float)window_R / ratio_k);
		max_axis = (int)max_axis;
		int left_pixel = (code_point_mat.at<float>(num, 1) - max_axis) < 0 ? 0 : (code_point_mat.at<float>(num, 1) - max_axis);
		int up_pixel = (code_point_mat.at<float>(num, 2) - max_axis) < 0 ? 0 : (code_point_mat.at<float>(num, 2) - max_axis);
		int right_pixel = (code_point_mat.at<float>(num, 1) + max_axis) >= ori_image.cols ? (ori_image.cols - 1) : (code_point_mat.at<float>(num, 1) + max_axis);
		int down_pixel = (code_point_mat.at<float>(num, 2) + max_axis) >= ori_image.rows ? (ori_image.rows - 1) : (code_point_mat.at<float>(num, 2) + max_axis);
		cv::Mat ori_image_part = ori_image(cv::Rect(left_pixel, up_pixel, right_pixel - left_pixel, down_pixel - up_pixel));
		Image2D_InterpCoef bic_var_cur(ori_image_part.cols, ori_image_part.rows);
		Eigen::MatrixXf Img_cur;
		cv::cv2eigen(ori_image_part, Img_cur);
		get_bicvar(Img_cur, &bic_var_cur);

		Eigen::Matrix3f Tr_s;
		Tr_s << code_point_mat.at<float>(num, 4) / re_scale_for_circle, 0, 0,
			0, code_point_mat.at<float>(num, 5) / re_scale_for_circle, 0,
			0, 0, 1;
		Eigen::Matrix3f Tr_r;
		Tr_r << cos(code_point_mat.at<float>(num, 6) + 1.57079632), sin(code_point_mat.at<float>(num, 6) + 1.57079632), 0,
			-sin(code_point_mat.at<float>(num, 6) + 1.57079632), cos(code_point_mat.at<float>(num, 6) + 1.57079632), 0,
			0, 0, 1;
		Eigen::Matrix3f Tr_t;
		Tr_t << 1, 0, code_point_mat.at<float>(num, 2) - up_pixel - (float)R_S - 0.5,
			0, 1, code_point_mat.at<float>(num, 1) - left_pixel - (float)R_S - 0.5,
			0, 0, 1;
		double angle_step = 6.28318531 / ((double)angle_search);

		float* param_optimal = new float[10];
		param_optimal[0] = 0;
		param_optimal[1] = 0;
		param_optimal[2] = 0;
		param_optimal[3] = 0;
		param_optimal[4] = 0;
		param_optimal[5] = 0;
		param_optimal[6] = 0;
		param_optimal[7] = 0;
		param_optimal[8] = 0;
		param_optimal[9] = 0;
		for (double theta = 0; theta < 6.28318531; theta += angle_step)
		{
			Eigen::Matrix3f Tr_rz;
			Tr_rz << cos(theta), -sin(theta), 0,
				sin(theta), cos(theta), 0,
				0, 0, 1;
			Eigen::Matrix3f Proj_matrix = Tr_t * Tr_r * Tr_s * Tr_rz;
			float* param = new float[10];
			param[0] = R_S;
			param[1] = R_S;
			param[2] = Proj_matrix(0, 2);
			param[3] = Proj_matrix(1, 2);
			param[4] = Proj_matrix(0, 0) - 1.0;
			param[5] = Proj_matrix(0, 1);
			param[6] = Proj_matrix(1, 0);
			param[7] = Proj_matrix(1, 1) - 1.0;
			param[8] = 0;
			param[9] = 0;
			int iter = 0;
			while (simple_DIC(param, Img_ref
				, hess, Cross_matrix
				, bic_var_cur, window_R) && iter < 5 && param[9]>1e-6)
			{
				iter++;
			}

			repeat_area = false;
			for (int pp = 0; pp < Code_inf_temp.size(); pp++)
			{
				float if_delta_x = ((param[0] + param[2] + up_pixel - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * Code_inf_temp[pp].sepckle_pos.dyp1
					- (param[1] + param[3] + left_pixel - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp1 + 1))
					/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
				float if_delta_y = -((param[0] + param[2] + up_pixel - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1)
					- (param[1] + param[3] + left_pixel - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp2))
					/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
				if (if_delta_x * if_delta_x + if_delta_y * if_delta_y < Code_inf_temp[pp].sepckle_pos.R * Code_inf_temp[pp].sepckle_pos.R)
				{
					repeat_area = true;
					break;
				}
			}
			if (!repeat_area && param[8] > param_optimal[8])
			{
				param_optimal[0] = param[0];
				param_optimal[1] = param[1];
				param_optimal[2] = param[2];
				param_optimal[3] = param[3];
				param_optimal[4] = param[4];
				param_optimal[5] = param[5];
				param_optimal[6] = param[6];
				param_optimal[7] = param[7];
				param_optimal[8] = param[8];
				param_optimal[9] = param[9];
			}
			delete[] param;
		}
		if (param_optimal[8] > min_cc_key_inital)
		{

			int iter = 0;
			while (simple_DIC(param_optimal, Img_ref
				, hess, Cross_matrix
				, bic_var_cur, window_R) && iter < 100 && param_optimal[9]>1e-10)
			{
				iter++;
			}
			if (param_optimal[8] > min_cc_key)
			{
				Pos_speckle pos_speckle_now(param_optimal[0], param_optimal[1], param_optimal[2] + up_pixel, param_optimal[3] + left_pixel,
					param_optimal[4], param_optimal[5], param_optimal[6], param_optimal[7], param_optimal[8], param_optimal[9], window_R);
				Code_speckle code_speckle_now(pos_circle_now, pos_speckle_now);
				decode_speckle(ori_image_part, bic_var_cur, left_pixel, up_pixel, code_speckle_now, ratio_k, min_cc_code, code_bites);
				if (code_speckle_now.code_value != -1)
				{
					code_speckle_now.circle_contour = contours_pixel[num];
					Code_inf_temp.push_back(code_speckle_now);
				}
			}
		}
		bic_var_cur.Delete2D_InterpCoef();
	}	
	bic_var_ref.Delete2D_InterpCoef();
	Code_inf.clear();
	for (int num = 0; num < Code_inf_temp.size(); num++)
	{
		Code_inf.push_back(Code_inf_temp[num]);
	}
	//cv::Mat II = cv::Mat::zeros(ori_image.size(), CV_8UC3);
	//cv::cvtColor(ori_image, II, CV_GRAY2BGR);

	//std::vector<std::vector<cv::Point>> contours_pixels;
	//for (int num = 0; num < Code_inf_temp.size(); num++)
	//{
	//	contours_pixels.push_back(Code_inf_temp[num].circle_contour);

	//	cv::putText(II, std::to_string(Code_inf_temp[num].code_value)
	//		, cv::Point2f(Code_inf_temp[num].sepckle_pos.dy+ Code_inf_temp[num].sepckle_pos.y
	//			, Code_inf_temp[num].sepckle_pos.dx + Code_inf_temp[num].sepckle_pos.x)
	//		, cv::FONT_HERSHEY_SIMPLEX, 2, CV_RGB(0, 255, 0), 4);

	//}
	////cv::drawContours(II, contours_pixel, -1, cv::Scalar(0, 0, 255), 10);
	//cv::drawContours(II, contours_pixels, -1, cv::Scalar(0, 255, 0), 10);
	//cv::namedWindow("cannys", cv::WINDOW_NORMAL);
	//cv::imshow("cannys", II);
	//cv::waitKey(0);
	return true;
}

//bool ImageDetectMethod::detectcodespeckle(cv::Mat ori_image, cv::Mat image_key, cv::Mat image_key_mask,
//	std::vector<Code_speckle>& Code_inf, float ratio_k, float min_cc_key, float min_cc_code, int code_bites,
//	float min_radius, float max_radius, float ellipse_error_pixel,
//	DetectContoursMethod image_process_method
//	, double max_aspect_ratio, int min_points, int min_contour_num)
//{
//	std::vector<std::vector<cv::Point>> contours_pixel;
//	cv::Mat code_point_mat;
//	detectcircle_rough(ori_image, code_point_mat, contours_pixel,
//		min_radius, max_radius, ellipse_error_pixel,
//		image_process_method, max_aspect_ratio, min_points, min_contour_num);
//
//	Eigen::MatrixXf Img_ref;
//	cv::cv2eigen(image_key, Img_ref);
//	int window_R = (image_key_mask.rows - 1) / 2;
//	int R_S = (image_key.rows - 1) / 2;
//
//	Eigen::MatrixXf Cross_matrix = Eigen::MatrixXf::Zero(window_R * 2 + 1, window_R * 2 + 1);
//	int duak_R = window_R * window_R;
//	int Num_weight = 0;
//	for (int pp = -window_R; pp < window_R + 1; ++pp)
//	{
//		for (int qq = -window_R; qq < window_R + 1; ++qq)
//		{
//			if ((pp * pp + qq * qq) <= duak_R)
//			{
//				Cross_matrix(pp + window_R, qq + window_R) = 1;
//				Num_weight++;
//			}
//			else
//			{
//				Cross_matrix(pp + window_R, qq + window_R) = 0;
//			}
//		}
//	}
//	Image2D_InterpCoef bic_var_ref(image_key.cols, image_key.rows);
//	Hess_struct hess;
//	get_bicvar(Img_ref, &bic_var_ref);
//
//	//计算Hess
//	get_hess_pixel(Img_ref, R_S, R_S, &bic_var_ref
//		, window_R, Cross_matrix, Num_weight, &hess);
//
//	int num = 0;
//	std::vector<Code_speckle> Code_inf_temp;
//	for (int num = 0; num < contours_pixel.size(); num++)
//	{
//		bool repeat_area = false;
//		for (int pp = 0; pp < Code_inf_temp.size(); pp++)
//		{
//			float if_delta_x = ((code_point_mat.at<float>(num, 2) - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * Code_inf_temp[pp].sepckle_pos.dyp1
//				- (code_point_mat.at<float>(num, 1) - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp1 + 1))
//				/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
//			float if_delta_y = -((code_point_mat.at<float>(num, 2) - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1)
//				- (code_point_mat.at<float>(num, 1) - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp2))
//				/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
//			if (if_delta_x * if_delta_x + if_delta_y * if_delta_y < Code_inf_temp[pp].sepckle_pos.R * Code_inf_temp[pp].sepckle_pos.R)
//			{
//				repeat_area = true;
//				break;
//			}
//		}
//		if (repeat_area)
//		{
//			continue;
//		}
//		Pos_ellispe pos_circle_now(code_point_mat.at<float>(num, 1), code_point_mat.at<float>(num, 2)
//			, code_point_mat.at<float>(num, 4), code_point_mat.at<float>(num, 5)
//			, code_point_mat.at<float>(num, 6), code_point_mat.at<float>(num, 3));
//		Eigen::MatrixXf Img_cur;
//		float max_axis = code_point_mat.at<float>(num, 4) > code_point_mat.at<float>(num, 5) ? code_point_mat.at<float>(num, 4) : code_point_mat.at<float>(num, 5);
//		max_axis = max_axis < 5 ? 5 : max_axis;
//		//max_axis *= (1.5 / ratio_k);
//		max_axis *= (1.5 * (float)R_S / (float)window_R / ratio_k);
//		max_axis = (int)max_axis;
//		int left_pixel = (code_point_mat.at<float>(num, 1) - max_axis) < 0 ? 0 : (code_point_mat.at<float>(num, 1) - max_axis);
//		int up_pixel = (code_point_mat.at<float>(num, 2) - max_axis) < 0 ? 0 : (code_point_mat.at<float>(num, 2) - max_axis);
//		int right_pixel = (code_point_mat.at<float>(num, 1) + max_axis) >= ori_image.cols ? (ori_image.cols - 1) : (code_point_mat.at<float>(num, 1) + max_axis);
//		int down_pixel = (code_point_mat.at<float>(num, 2) + max_axis) >= ori_image.rows ? (ori_image.rows - 1) : (code_point_mat.at<float>(num, 2) + max_axis);
//		cv::Mat ori_image_part = ori_image(cv::Rect(left_pixel, up_pixel, right_pixel - left_pixel, down_pixel - up_pixel));
//		Image2D_InterpCoef bic_var_cur(ori_image_part.cols, ori_image_part.rows);
//		cv::cv2eigen(ori_image_part, Img_cur);
//		get_bicvar(Img_cur, &bic_var_cur);
//
//		Eigen::Matrix3f Tr_s;
//		Tr_s << code_point_mat.at<float>(num, 4) / ((float)R_S + (float)window_R) * 2.0, 0, 0,
//			0, code_point_mat.at<float>(num, 5) / ((float)R_S + (float)window_R) * 2.0, 0,
//			0, 0, 1;
//		Eigen::Matrix3f Tr_r;
//		Tr_r << cos(code_point_mat.at<float>(num, 6) + 1.57079632), sin(code_point_mat.at<float>(num, 6) + 1.57079632), 0,
//			-sin(code_point_mat.at<float>(num, 6) + 1.57079632), cos(code_point_mat.at<float>(num, 6) + 1.57079632), 0,
//			0, 0, 1;
//		Eigen::Matrix3f Tr_t;
//		Tr_t << 1, 0, code_point_mat.at<float>(num, 2) - up_pixel - (float)R_S - 0.5,
//			0, 1, code_point_mat.at<float>(num, 1) - left_pixel - (float)R_S - 0.5,
//			0, 0, 1;
//		double angle_step = 6.28318531 / (12.0);
//
//		float* param_optimal = new float[10];
//		float min_ss0 = -1;
//		param_optimal[0] = 0;
//		param_optimal[1] = 0;
//		param_optimal[2] = 0;
//		param_optimal[3] = 0;
//		param_optimal[4] = 0;
//		param_optimal[5] = 0;
//		param_optimal[6] = 0;
//		param_optimal[7] = 0;
//		param_optimal[8] = 0;
//		param_optimal[9] = 0;
//		int iter_max = 0;
//		for (double theta = 0; theta < 6.28318531; theta += angle_step)
//		{
//			Eigen::Matrix3f Tr_rz;
//			Tr_rz << cos(theta), -sin(theta), 0,
//				sin(theta), cos(theta), 0,
//				0, 0, 1;
//			//Eigen::Matrix3f Tr_rz;
//			//Tr_rz << cos(theta), -sin(theta), (1-cos(theta))* R_S+sin(theta)*R_S,
//			//	sin(theta), cos(theta), (1 - cos(theta))* R_S - sin(theta) * R_S,
//			//	0, 0, 1;
//			Eigen::Matrix3f Proj_matrix = Tr_t * Tr_r * Tr_s * Tr_rz;
//			float* param = new float[10];
//			param[0] = R_S;
//			param[1] = R_S;
//			param[2] = Proj_matrix(0, 2);
//			param[3] = Proj_matrix(1, 2);
//			param[4] = Proj_matrix(0, 0) - 1.0;
//			param[5] = Proj_matrix(0, 1);
//			param[6] = Proj_matrix(1, 0);
//			param[7] = Proj_matrix(1, 1) - 1.0;
//			param[8] = 0;
//			param[9] = 0;
//			int iter = 0;
//			while (simple_DIC(param, Img_ref
//				, hess, Cross_matrix
//				, bic_var_cur, window_R) && iter < 1 && param[9]>1e-6)
//			{
//				if (iter == 0)
//				{
//					min_ss0 = min_ss0 < param[8] ? param[8] : min_ss0;
//				}
//				iter++;
//			}
//			repeat_area = false;
//			for (int pp = 0; pp < Code_inf_temp.size(); pp++)
//			{
//				float if_delta_x = ((param[0] + param[2] + up_pixel - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * Code_inf_temp[pp].sepckle_pos.dyp1
//					- (param[1] + param[3] + left_pixel - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp1 + 1))
//					/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
//				float if_delta_y = -((param[0] + param[2] + up_pixel - Code_inf_temp[pp].sepckle_pos.x - Code_inf_temp[pp].sepckle_pos.dx) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1)
//					- (param[1] + param[3] + left_pixel - Code_inf_temp[pp].sepckle_pos.y - Code_inf_temp[pp].sepckle_pos.dy) * (Code_inf_temp[pp].sepckle_pos.dxp2))
//					/ ((Code_inf_temp[pp].sepckle_pos.dxp2) * (Code_inf_temp[pp].sepckle_pos.dyp1) - (Code_inf_temp[pp].sepckle_pos.dxp1 + 1) * (Code_inf_temp[pp].sepckle_pos.dyp2 + 1));
//				if (if_delta_x * if_delta_x + if_delta_y * if_delta_y < Code_inf_temp[pp].sepckle_pos.R * Code_inf_temp[pp].sepckle_pos.R)
//				{
//					repeat_area = true;
//					break;
//				}
//			}
//			std::cout << param[8] << std::endl;
//			if (!repeat_area && param[8] > param_optimal[8])
//			{
//				param_optimal[0] = param[0];
//				param_optimal[1] = param[1];
//				param_optimal[2] = param[2];
//				param_optimal[3] = param[3];
//				param_optimal[4] = param[4];
//				param_optimal[5] = param[5];
//				param_optimal[6] = param[6];
//				param_optimal[7] = param[7];
//				param_optimal[8] = param[8];
//				param_optimal[9] = param[9];
//				iter_max = iter;
//			}
//			delete[] param;
//		}
//		if (param_optimal[8] > min_cc_key)
//		{
//			int iter = 0;
//			while (simple_DIC(param_optimal, Img_ref
//				, hess, Cross_matrix
//				, bic_var_cur, window_R) && iter < 200 && param_optimal[9]>1e-6)
//			{
//				if (iter == 0)
//				{
//					min_ss0 = min_ss0 < param_optimal[8] ? param_optimal[8] : min_ss0;
//				}
//				iter++;
//			}
//			Pos_speckle pos_speckle_now(param_optimal[0], param_optimal[1], param_optimal[2] + up_pixel, param_optimal[3] + left_pixel,
//				param_optimal[4], param_optimal[5], param_optimal[6], param_optimal[7], param_optimal[8], param_optimal[9], R_S);
//			Code_speckle code_speckle_now(pos_circle_now, pos_speckle_now);
//			decode_speckle(ori_image_part, bic_var_cur, left_pixel, up_pixel, code_speckle_now, ratio_k, min_cc_code, code_bites);
//			if (code_speckle_now.code_value != -1)
//			{
//				code_speckle_now.circle_contour = contours_pixel[num];
//				Code_inf_temp.push_back(code_speckle_now);
//			}
//		}
//
//		bic_var_cur.Delete2D_InterpCoef();
//	}
//	bic_var_ref.Delete2D_InterpCoef();
//	Code_inf.clear();
//	for (int num = 0; num < Code_inf_temp.size(); num++)
//	{
//		Code_inf.push_back(Code_inf_temp[num]);
//	}
//	//cv::Mat II = cv::Mat::zeros(ori_image.size(), CV_8UC3);
//	//cv::cvtColor(ori_image, II, CV_GRAY2BGR);
//
//	//std::vector<std::vector<cv::Point>> contours_pixels;
//	//for (int num = 0; num < Code_inf_temp.size(); num++)
//	//{
//	//	contours_pixels.push_back(Code_inf_temp[num].circle_contour);
//
//	//	cv::putText(II, std::to_string(Code_inf_temp[num].code_value)
//	//		, cv::Point2f(Code_inf_temp[num].sepckle_pos.dy+ Code_inf_temp[num].sepckle_pos.y
//	//			, Code_inf_temp[num].sepckle_pos.dx + Code_inf_temp[num].sepckle_pos.x)
//	//		, cv::FONT_HERSHEY_SIMPLEX, 2, CV_RGB(0, 255, 0), 4);
//
//	//}
//	////cv::drawContours(II, contours_pixel, -1, cv::Scalar(0, 0, 255), 10);
//	//cv::drawContours(II, contours_pixels, -1, cv::Scalar(0, 255, 0), 10);
//	//cv::namedWindow("cannys", cv::WINDOW_NORMAL);
//	//cv::imshow("cannys", II);
//	//cv::waitKey(0);
//	return true;
//}
bool ImageDetectMethod::detectcircle_rough(cv::Mat ori_image, cv::Mat& code_point_mat
	, std::vector<std::vector<cv::Point>>& contours_pixel,
	float min_radius, float max_radius, float ellipse_error_pixel /*=0.5*/,
	DetectContoursMethod image_process_method /*= CANNY_Method*/,
	double max_aspect_ratio, int min_points, int min_contour_num)
{
	if (!ori_image.data)        // 判断图片调入是否成功
		return false;        // 调入图片失败则退出

	if (ori_image.channels() == 3)
	{
		cv::cvtColor(ori_image, ori_image, CV_BGR2GRAY);
	}
	contours_pixel.clear();
	QList<QList<float>> ellipse_pars_all;
	cv::Mat processed_image_mat;
	ImagePreprocess(ori_image, processed_image_mat);

	std::vector<std::vector<cv::Point>> contours;
	QList<QList<float>> ellipse_pars;
	DetectClosedContours(processed_image_mat, contours, image_process_method);
	FilterEllipseContours(contours, min_radius, max_radius,
		ellipse_error_pixel, ellipse_pars, max_aspect_ratio, min_points, min_contour_num);
	
	for (int i = 0; i < ellipse_pars.size(); i++)
	{
		ellipse_pars_all.append(ellipse_pars[i]);
		contours_pixel.push_back(contours[ellipse_pars[i][6]]);
	}
	//ellipse_pars - n*6  center_x,center_y,r_a,r_b,angle_inPI,ellipse_error,contours_index,ID,code_type(0- uncode,1- code)
	code_point_mat = cv::Mat();
	for (int i = 0; i < ellipse_pars_all.size(); i++)
	{
		float a[7] = { 0,ellipse_pars_all[i][0],ellipse_pars_all[i][1],ellipse_pars_all[i][5],
			ellipse_pars_all[i][2],ellipse_pars_all[i][3],ellipse_pars_all[i][4] };
		cv::Mat mat = cv::Mat(1, 7, CV_32F, a);
		code_point_mat.push_back(mat);
	}
	return true;
}



bool ImageDetectMethod::simple_DIC(float* param, Eigen::MatrixXf Img_refs
	, Hess_struct Hess, Eigen::MatrixXf Cross_matrix
	, Image2D_InterpCoef B_cof, int R)
{
	double mean_val = 0;
	double val_num = 0;
	int point_x = param[0];
	int point_y = param[1];
	Eigen::MatrixXf Img_ref(2 * R + 1, 2 * R + 1);
	Eigen::MatrixXf Img_cur(2 * R + 1, 2 * R + 1);
	float F[6];
	float V[6];
	for (int ii = point_x - R; ii < (point_x + R + 1); ii++)
	{
		for (int jj = point_y - R; jj < (point_y + R + 1); jj++)
		{

			int index_x = ii - point_x + R;
			int index_y = jj - point_y + R;

			double delta_x = ii - point_x;
			double delta_y = jj - point_y;
			if (Cross_matrix(index_x, index_y))
			{
				double tran_x = ii + param[2] + param[4] * delta_x + param[5] * delta_y;
				double tran_y = jj + param[3] + param[6] * delta_x + param[7] * delta_y;
				int tran_x_floor = floor(tran_x);
				int tran_y_floor = floor(tran_y);
				double tran_x_delta = tran_x - tran_x_floor;
				double tran_y_delta = tran_y - tran_y_floor;
				if (tran_x_floor < 0 || tran_x_floor >= B_cof.height || tran_y_floor < 0 || tran_y_floor >= B_cof.width)
				{
					return false;
				}

				F[1] = tran_x_delta;
				F[2] = F[1] * tran_x_delta;
				F[3] = F[2] * tran_x_delta;
				F[4] = F[3] * tran_x_delta;
				F[5] = F[4] * tran_x_delta;
				V[1] = tran_y_delta;
				V[2] = V[1] * tran_y_delta;
				V[3] = V[2] * tran_y_delta;
				V[4] = V[3] * tran_y_delta;
				V[5] = V[4] * tran_y_delta;
				Img_cur(index_x, index_y) =
					(B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(0, 5) * V[5])
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(1, 5) * V[5]) * F[1]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(2, 5) * V[5]) * F[2]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(3, 5) * V[5]) * F[3]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(4, 5) * V[5]) * F[4]
					+ (B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 0) + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 1) * V[1] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 2) * V[2] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 3) * V[3] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 4) * V[4] + B_cof.eg_mat[tran_x_floor][tran_y_floor].InterpCoef_mat(5, 5) * V[5]) * F[5];

				mean_val = mean_val + Img_cur(index_x, index_y);
				Img_ref(index_x, index_y) = Img_refs(ii, jj);
				val_num = val_num + 1;
			}

		}
	}
	//cv::Mat mat_23f;
	//Eigen::MatrixXf Img_reff = Img_ref / 255;
	//cv::eigen2cv(Img_reff, mat_23f);
	//cv::Mat mat_24f;
	//Eigen::MatrixXf Img_curr = Img_cur / 255;
	//cv::eigen2cv(Img_curr, mat_24f);
	mean_val = mean_val / val_num;
	double dev_val = 0;
	for (int ii = point_x - R; ii < (point_x + R + 1); ii++)
	{
		for (int jj = point_y - R; jj < (point_y + R + 1); jj++)
		{

			int index_x = ii - point_x + R;
			int index_y = jj - point_y + R;

			double delta_x = ii - point_x;
			double delta_y = jj - point_y;

			if (Cross_matrix(index_x, index_y))
			{
				dev_val = dev_val + pow((Img_cur(index_x, index_y) - mean_val), 2);
			}
		}
	}
	dev_val = sqrt(dev_val);
	if (dev_val > 0)
	{
		dev_val = 1 / dev_val;
	}
	else
	{
		return false;
	}

	Eigen::MatrixXf grad_cof = Eigen::MatrixXf::Zero(6, 1);
	param[8] = 0;

	for (int ii = point_x - R; ii < (point_x + R + 1); ii++)
	{
		for (int jj = point_y - R; jj < (point_y + R + 1); jj++)
		{

			int index_x = ii - point_x + R;
			int index_y = jj - point_y + R;

			double delta_x = ii - point_x;
			double delta_y = jj - point_y;

			if (Cross_matrix(index_x, index_y))
			{
				double norm_diff = (Img_ref(index_x, index_y) - Hess.mean_val) * Hess.dev_val - (Img_cur(index_x, index_y) - mean_val) * dev_val;
				double norm_diff_CC = (Img_ref(index_x, index_y) - Hess.mean_val) * (Img_cur(index_x, index_y) - mean_val);

				grad_cof(0, 0) += + norm_diff * Hess.Dif_inf.Dv(index_x, index_y);
				grad_cof(1, 0) += + norm_diff * Hess.Dif_inf.Du(index_x, index_y);
				grad_cof(2, 0) += + norm_diff * Hess.Dif_inf.Dvp1(index_x, index_y);
				grad_cof(3, 0) += + norm_diff * Hess.Dif_inf.Dvp2(index_x, index_y);
				grad_cof(4, 0) += norm_diff * Hess.Dif_inf.Dup1(index_x, index_y);
				grad_cof(5, 0) += norm_diff * Hess.Dif_inf.Dup2(index_x, index_y);
				param[8] += norm_diff_CC;
			}
		}
	}
	param[8] *= (Hess.dev_val * dev_val);
	for (int qq = 0; qq < 6; qq++)
	{
		grad_cof(qq) *= -2 * Hess.dev_val;
	}
	auto delta_params = Hess.Hess_mat.fullPivHouseholderQr().solve(grad_cof);
	param[9] = sqrt(pow(delta_params(0, 0), 2) + pow(delta_params(1, 0), 2) + pow(delta_params(2, 0), 2) + pow(delta_params(3, 0), 2) + pow(delta_params(4, 0), 2) + pow(delta_params(5, 0), 2));
	double denominator = delta_params(5, 0) + delta_params(2, 0) + delta_params(4, 0) * delta_params(1, 0) - delta_params(4, 0) * delta_params(3, 0) + 1;
	if (denominator == 0)
	{
		return false;
	}
	double add_v = param[2];
	double add_u = param[3];
	double add_dvp1 = param[4];
	double add_dvp2 = param[5];
	double add_dup1 = param[6];
	double add_dup2 = param[7];
	param[2] = add_v - ((add_dvp1 + 1) * (delta_params(0, 0) * (1 + delta_params(5, 0)) - delta_params(1, 0) * delta_params(3, 0)) +
		add_dvp2 * (delta_params(1, 0) * (1 + delta_params(2, 0)) - delta_params(0, 0) * delta_params(4, 0))) / denominator;
	param[3] = add_u - ((add_dup2 + 1) * (delta_params(1, 0) * (1 + delta_params(2, 0)) - delta_params(0, 0) * delta_params(4, 0)) + 
		add_dup1 * (delta_params(0, 0) * (1 + delta_params(5, 0)) - delta_params(1, 0) * delta_params(3, 0))) / denominator;
	param[4] = ((delta_params(5, 0) + 1) * (add_dvp1 + 1) - delta_params(4, 0) * add_dvp2) / denominator - 1;
	param[5] = (add_dvp2 * (delta_params(2, 0) + 1) - delta_params(3, 0) * (add_dvp1 + 1)) / denominator;
	param[6] = (add_dup1 * (delta_params(5, 0) + 1) - delta_params(4, 0) * (add_dup2 + 1)) / denominator;
	param[7] = ((delta_params(2, 0) + 1) * (add_dup2 + 1) - delta_params(3, 0) * add_dup1) / denominator - 1;
	return true;


}