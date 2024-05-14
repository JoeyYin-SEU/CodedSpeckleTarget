#ifndef IMAGEDETECTMETHOD_H
#define IMAGEDETECTMETHOD_H


/********************
Originally written by Professor Dong Shuai, with extensive detailed modifications by Yin Zhuoyi.
*******************/
#include <QObject>
#include "CMMHelp.h"
#include "CodeID.h"
#include "Eigen/Eigen"
#include<opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include<vector>
#include <fstream>
#include <algorithm>
#include<ceres/ceres.h>
#include <ceres/problem.h>
#include "Matlab2C/rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include "Matlab2C/imedge_2d.h"
#include "Matlab2C/get_chessborad_pixel.h"
#include "Matlab2C/get_fx_fy.h"
#include "Matlab2C/get_chessborad_pixel_terminate.h"
#include "Matlab2C/rt_nonfinite.h"
#include "Matlab2C/coder_array.h"
#include "Matlab2C/get_resize.h"
#include <fftw3.h>

typedef Eigen::Matrix<float, 6, 6> Matrix6f;


struct Pos_ellispe
{
	float centre_x;
	float centre_y;
	float r_a;
	float r_b;
	float theta;
	float error;
	Pos_ellispe(float centre_xs, float centre_ys, float r_as, float r_bs, float thetas, float errors)
		:centre_x(centre_xs), centre_y(centre_ys), r_a(r_as), r_b(r_bs), theta(thetas), error(errors)
	{
	}
	Pos_ellispe()
	{
	}
};
struct Pos_speckle
{
	float x;
	float y;
	float dx;
	float dy;
	float dxp1;
	float dxp2;
	float dyp1;
	float dyp2;
	float cc;
	float stop_grad;
	float R;
	Pos_speckle(float xs, float ys, float dxs, float dys, float dxp1s, float dxp2s, float dyp1s, float dyp2s, float ccs, float stop_grads, float Rs)
		:x(xs), y(ys), dx(dxs), dy(dys), dxp1(dxp1s), dxp2(dxp2s), dyp1(dyp1s), dyp2(dyp2s),cc(ccs), stop_grad(stop_grads), R(Rs)
	{
	}
	Pos_speckle()
	{
	}
};
struct Code_speckle
{
	int code_value;
	Pos_ellispe circle_pos;
	Pos_speckle sepckle_pos;
	std::vector<cv::Point> circle_contour;
	Code_speckle(Pos_ellispe circle_poss, Pos_speckle sepckle_poss)
		:circle_pos(circle_poss), sepckle_pos(sepckle_poss)
	{
		code_value = -1;
		circle_contour = {};
	}
	Code_speckle()
	{
	}
};


struct Dif_DIC
{
	Eigen::MatrixXf Dv;
	Eigen::MatrixXf Du;
	Eigen::MatrixXf Dvp1;
	Eigen::MatrixXf Dvp2;
	Eigen::MatrixXf Dup1;
	Eigen::MatrixXf Dup2;
};

struct Hess_struct
{
	Matrix6f Hess_mat;
	bool valid;
	float mean_val;
	float dev_val;
	Dif_DIC Dif_inf;
	Hess_struct()
	{
		valid = false;
	}
};

struct InterpCoef_struct
{
	Matrix6f InterpCoef_mat;
	bool valid;
	InterpCoef_struct()
	{
		valid = false;
	}
};
class Image2D_InterpCoef
{
public:
	int height, width;
	InterpCoef_struct** eg_mat = nullptr;
	Image2D_InterpCoef();
	Image2D_InterpCoef(int width, int height);
	void Delete2D_InterpCoef();
	~Image2D_InterpCoef();
};

struct Contour
{
	std::vector<cv::Point2f> points;
	std::vector<float> direction;
	std::vector<float> response;
};

class ImageDetectMethod : public QObject
{
	Q_OBJECT

public:
	ImageDetectMethod(QObject* parent);
	~ImageDetectMethod();

	//CSI标定板
	static bool FindCircleGrid(cv::Mat ori_image, int h_num, int v_num, int h_offset, int v_offset, int h_mid_length,
		int v_mid_length, std::vector<cv::Point2f>& corners, std::vector<uchar>& sign_list, int& useful_corner_num,
		std::vector<std::vector<cv::Point2f>>& edge_points,
		float max_ratio = 3, float ratio = 0.5,
		float min_radius = 5, float max_radius = 50, float ellipse_error_pixel = 0.5 , int min_arc = 240, int min_points = 6, int min_contour_num = 8,
		DetectContoursMethod image_process_method = CANNY_Method, SubPixelPosMethod subpixel_pos_method = SubPixelPosMethod::SubPixel_Interpolation);

	//棋盘标定板角点识别
	static bool detectCheckerboard(cv::Mat I, std::vector<cv::Point2f>& corner_points, cv::Size& wh_check, 
		double peakThreshold = 0.15, bool highDistortion = false, bool usePartial = false);
	
	static bool detectcodecircle(cv::Mat ori_image, cv::Mat& code_point_mat, std::vector<std::vector<cv::Point2f>> & contours_pixel,
		float ratio_k, float ratio_k1, float ratio_k2,
		float min_radius, float max_radius, float ellipse_error_pixel = 0.5,
		MarkPointColorType color_type = BlackDownWhiteUp, CodePointBitesType code_bites_type = CodeBites15,
		DetectContoursMethod image_process_method = CANNY_Method, SubPixelPosMethod subpixel_pos_method = Gray_Centroid
		, double max_aspect_ratio = 2, int min_points = 6, int min_contour_num = 8,
		float delta_Mt = 50, float fore_stdDev = 100, float back_stdDev = 100);

	static bool detectcircle_rough(cv::Mat ori_image, cv::Mat& code_point_mat, std::vector<std::vector<cv::Point>>& contours_pixel,
		float min_radius, float max_radius, float ellipse_error_pixel = 0.5,
		DetectContoursMethod image_process_method = CANNY_Method
		, double max_aspect_ratio = 2, int min_points = 6, int min_contour_num = 8);

	static bool detectcodespeckle(cv::Mat ori_image, cv::Mat image_key, cv::Mat image_key_mask1, cv::Mat image_key_mask2,
		std::vector<Code_speckle> &Code_inf, float ratio_k, float min_cc_key_inital=0.4, float min_cc_key = 0.5, float min_cc_code = 0.5, int code_bites = 10,
		int temple_R = 50, int angle_search = 12, float min_radius = 10, float max_radius = 100, float ellipse_error_pixel = 0.5,
		DetectContoursMethod image_process_method = CANNY_Method
		, double max_aspect_ratio = 2, int min_points = 6, int min_contour_num = 8);

	static bool simple_DIC(float *param, Eigen::MatrixXf Img_refs
		, Hess_struct Hess, Eigen::MatrixXf Cross_matrix
		, Image2D_InterpCoef B_cof, int R);
	

private:
	//图像预处理
	static inline double getAmplitude(cv::Mat& dx, cv::Mat& dy, int i, int j);
	static inline void getMagNeighbourhood(cv::Mat& dx, cv::Mat& dy, cv::Point& p, int w, int h, std::vector<double>& mag);
	static inline void get2ndFacetModelIn3x3(std::vector<double>& mag, std::vector<double>& a);
	static inline void eigenvals(std::vector<double>& a, double eigval[2], double eigvec[2][2]);
	static inline double vector2angle(double x, double y);
	static cv::Mat Image_Resize(const cv::Mat& ori_image_mat, int hei, int wid, int method,bool anti);
	static bool ImagePreprocess(const cv::Mat& ori_image_mat, cv::Mat& processed_image_mat);
	static bool DetectClosedContours(const cv::Mat& image_mat, std::vector<std::vector<cv::Point>>& contours, DetectContoursMethod image_process_method);
	static void Sub_pixel_edge(const cv::Mat image_mat, std::vector<std::vector<cv::Point2f>>& Cons, double sigma = sqrt(2)
		, DetectContoursMethod type = CANNY_Method);
	static void extractSubPixPoints(cv::Mat& dx, cv::Mat& dy, std::vector<std::vector<cv::Point> >& contoursInPixel
		, std::vector<std::vector<cv::Point2f>>& contours);

	//ellipse_pars - n*6  center_x,center_y,r_a,r_b,angle_inPI,ellipse_error,contours_index,ID
	static bool FilterEllipseContours(const std::vector<std::vector<cv::Point>>& contours, int min_radius_pixel, int max_radius_pixel, float ellipse_error_pixel,
		QList<QList<float>>& ellipse_pars, double max_aspect_ratio = 2, int min_points = 6, int min_contour_num = 8
		, int min_arc = 240);

	static float Contours_Length(std::vector<cv::Point> C);
	static float Contours_arc(std::vector<cv::Point> C, cv::Point2f center);

	static float ErrorDROfEllipseFit(float center_x, float center_y, float ellipse_a, float ellipse_b, float ellipse_angle,
		float x, float  y);
	static float LeastSquareErrorOfEllipseFit(float center_x, float center_y, float ellipse_a, float ellipse_b, float ellipse_angle,
		float x, float  y);

	//进一步筛选出适用于编码点解码的圆点
	static bool FilterEllipseContoursForCodePoint(const cv::Mat& image_mat, float ratio_k, float ratio_k1, float ratio_k2,
		QList<QList<float>>& ellipse_pars,
		float delta_Mt = 50, float fore_stdDev = 100, float back_stdDev = 100);

	static bool EllipseGrayJudgeForPointCSI(const cv::Mat& image_mat, float center_x, float center_y,
		float ellipse_a, float ellipse_b, float angle_in_pi, float ratio_k,
		float& out_std);
	static bool EllipseGrayJudgeForPointCSI_is2Circle(
		QList<QList<float>> ellipse_pars_all, QList<float> ellipse_pars_now, float rati_k);

	static bool EllipseGrayJudgeForCodePoint(const cv::Mat& image_mat, float center_x, float center_y,
		float ellipse_a, float ellipse_b, float angle_in_pi, float ratio_k,
		float& out_foreground_mean, float& out_background_mean, float& out_foreground_stdDev, float& out_background_stdDev,
		float delta_Mt = 50, float fore_stdDev = 100, float back_stdDev = 100);

	//获取编码值数组的指针，
	static int* ReturnDefualtIdArray(int& array_size, CodePointBitesType code_bites_type = CodeBites15);

	//解码,编码点解码
	static bool Decoding20140210(const cv::Mat& image_mat, int& out_put_code_id, float center_x, float center_y, float ellipse_a, float ellipse_b, float angle_in_pi,
		float ratio_k1 = 2.4, float ratio_k2 = 4, MarkPointColorType color_type = BlackDownWhiteUp, CodePointBitesType code_bites_type = CodeBites15,
		double thread_value = 0);

	static bool CalculateRealCodeID20140210(QList<int> in_put_code_list, QList<int>& out_put_code_list, int& out_put_code_ID);
	//非编码点判断
	static bool UncodePointCheck(const cv::Mat& image_mat, float center_x, float center_y, float ellipse_a, float ellipse_b, float angle_in_pi,
		float ratio_k = 2, MarkPointColorType color_type = BlackDownWhiteUp, CodePointBitesType code_bites_type = CodeBites15);
	
	//************根据拟合误差，筛选部分点重新选点进行拟合
	static void ReduceBadEllipseFitPoints(std::vector<cv::Point2f>& edge_points, float center_x, float center_y, float ellipse_a, float ellipse_b, float angle_in_pi);

	static float CalWeightOfCentriod(float gray_value, float gray_threshold,
		MarkPointColorType color_type = BlackDownWhiteUp,
		SubPixelPosMethod subPixel_method = Gray_Centroid);
	static bool CalCentriodBySubsetMat(const cv::Mat& subset_mat, float gray_threshold, float& sub_center_x, float& sub_center_y,
		MarkPointColorType color_type = BlackDownWhiteUp,
		SubPixelPosMethod subPixel_method = Gray_Centroid);
	//亚像素定位
	static bool FindSubPixelPosOfCircleCenter(const cv::Mat& image_mat, float center_x, float center_y, float ellipse_a, float ellipse_b,
		float angle_in_pi, const std::vector<cv::Point>& contour_points,
		float& sub_pixel_center_x, float& sub_pixel_center_y, std::vector<cv::Point2f>& subpixel_edge_points
		, SubPixelPosMethod subpixel_pos_method = SubPixelPosMethod::Gray_Centroid);

	//获取椭圆局部感兴趣区域
	static QRect GetEllipseROIRect(const cv::Mat& image_mat, float center_x, float center_y, float ellipse_a, float ellipse_b, float angle_in_pi);


	//******如果没有指定，黑底白点或白底黑点时，判断是哪种类型
	static MarkPointColorType JudgeTargetColorType(const cv::Mat& sub_mat, float center_x_insubmat, float center_y_insubmat, float ellipse_a, float ellipse_b, float angle_in_pi);



	//*********************获取由椭圆圆心发出射线与编码环两个交点之间的线段上的灰度值******************
	//image_mat-输入原始图像
	//point1-输入交点1坐标
	//point2-输入交点2坐标
	//返回一系列数
	static QList<int> GetALineGrayList(cv::Mat image_mat, QPoint point1, QPoint point2);

	//**************平均值****
	static int AverageOfList(QList<int>& list_value);

	static double MeanValue(QList<float> value_list);
	//*************************获取一系列数的中值*********************************
	//value_list - 输入数组
	//返回中值
	static int MIdValue(QList<int> value_list);

	//***********二进制转十进制，输入list_code2二进制数，输出return十进制值**********************
	static int Change2To10(QList<int> list_code2);

	static bool FilterEllipseContours_for_distance(std::vector<std::vector<cv::Point>> contours, QList<QList<float>>& ellipse_pars);

	//筛选出CSI类圆标定板的 圆点方向点（外圆点）+ 普通点， vector<int> orident_point_index,3个方向点的index
	static bool FilterEllipseContoursForCSICalibrationPlane(const cv::Mat& image_mat,
		QList<QList<float>>& ellipse_pars_all,
		QList<QList<float>>& ellipse_pars, QList<QList<float>>& ellipse_pars_ori, float ratio_k = 0.5);

	//计算直线方程Ax+By+C=0
	static void LineEquation(cv::Point2f p1, cv::Point2f p2, float& A, float& B, float& C);

	//计算点p到直线Ax+By+C=0距离d
	static float PointTpLineDistance(cv::Point2f p, float A, float B, float C);

	//计算点到点之间距离
	static float PointToPointDistance(cv::Point2f p1, cv::Point2f p2);

	//DIC计算插值矩阵
	static void get_bicvar(Eigen::MatrixXf value_mat, Image2D_InterpCoef* bic_var);

	//计算Hess
	static void get_hess_pixel(Eigen::MatrixXf value_mat, int pixel_x, int pixel_y, Image2D_InterpCoef* bic_var
		, int window_R, Eigen::MatrixXf Cross_matrix, int Num_weight, Hess_struct* hess);

	//解码散斑
	static void decode_speckle(cv::Mat part, Image2D_InterpCoef B_cof, int left_b, int up_b, Code_speckle &Code_inf, float ratio_k, float min_cc = 0.7, int code_bites = 10);
	
	//计算归一化相关
	static float cal_cc(std::vector<float> v1, float mean_v1, float invstd_v1, std::vector<float> v2, float mean_v2, float invstd_v2);

};

#endif // IMAGEDETECTMETHOD_H
