// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "KKH_NMS_NMS_Greedy.h"
#include "JNI_type_converter.h"

#define ARMA_64BIT_WORD
#include "armadillo"

void merge_dets(const arma::Mat<float> &dr, const arma::Col<float> &ds,
	arma::Mat<float> &dr_new, arma::Col<float> &ds_new, float overlap_thresh);

JNIEXPORT void JNICALL Java_KKH_NMS_NMS_1Greedy_suppress
(JNIEnv *env, jobject jobj, jfloatArray dr_, jfloatArray ds_, jfloat overlap_thresh_)
{
	jni_wrap jw(env, jobj);
	std::vector<float> dr = jw.from_jfloatArray(dr_);
	std::vector<float> ds = jw.from_jfloatArray(ds_);
	float overlap_thresh = overlap_thresh_;

	int ndr = ds.size();

	arma::Mat<float> dr_arma(dr.data(), ndr, 4, false, false);
	arma::Col<float> ds_arma(ds.data(), ndr, false, false);
	arma::Mat<float> dr_new;
	arma::Col<float> ds_new;

	merge_dets(dr_arma, ds_arma, dr_new, ds_new, overlap_thresh);

	int ndr_nms = dr_new.n_rows;

	jw.set_field<jfloatArray>("dr_nms", jw.to_jfloatArray(dr_new.memptr(), ndr_nms * 4));
	jw.set_field<jfloatArray>("ds_nms", jw.to_jfloatArray(ds_new.memptr(), ndr_nms));

}

void merge_dets(const arma::Mat<float> &dr, const arma::Col<float> &ds,
	arma::Mat<float> &dr_new, arma::Col<float> &ds_new, float overlap_thresh)
{
	dr_new.set_size(0, 0);
	ds_new.set_size(0);
		
	if (dr.n_rows == 0) return;

	arma::Col<float> x1 = dr.col(0);
	arma::Col<float> y1 = dr.col(1);
	arma::Col<float> x2 = dr.col(0) + dr.col(2);
	arma::Col<float> y2 = dr.col(1) + dr.col(3);
	arma::Col<float> s = ds;

	arma::Col<float>area = (x2 - x1 + 1) % (y2 - y1 + 1);
	arma::uvec I = arma::sort_index(s);
	arma::Col<float> vals = s(I);

	arma::uvec pick = arma::zeros<arma::uvec>(s.n_elem);
	unsigned int counter = 0;

	arma::Col<float> temp1, temp2, w, h, o, xx1, xx2, yy1, yy2;
	arma::uvec I_sub;

	while (I.n_elem > 0)
	{
		int last = I.n_elem;
		unsigned int i = I(last - 1);
		pick(counter) = i;
		counter++;

		if (last == 1) break;

		I_sub = I.subvec(0, last - 2);

		temp1 = x1(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(x1(i)), 1, 1), temp1.n_elem, 1);
		xx1 = arma::max(temp2, temp1);

		temp1 = x2(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(x2(i)), 1, 1), temp1.n_elem, 1);
		xx2 = arma::min(temp2, temp1);

		temp1 = y1(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(y1(i)), 1, 1), temp1.n_elem, 1);
		yy1 = arma::max(temp2, temp1);

		temp1 = y2(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(y2(i)), 1, 1), temp1.n_elem, 1);
		yy2 = arma::min(temp2, temp1);

		temp1 = xx2 - xx1 + 1;
		temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
		w = arma::max(temp2, temp1);

		temp1 = yy2 - yy1 + 1;
		temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
		h = arma::max(temp2, temp1);

		o = w % h / area(I_sub);
		I = I(arma::find(o <= overlap_thresh));

	}

	pick = pick.subvec(0, counter - 1);
	dr_new = dr.rows(pick);
	ds_new = ds(pick);
}
