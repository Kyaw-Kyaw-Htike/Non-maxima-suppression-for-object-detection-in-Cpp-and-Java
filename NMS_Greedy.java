package KKH.NMS;

/**
 * Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
 */
public class NMS_Greedy {

    private float[] dr_nms;
    private float[] ds_nms;

    static {System.loadLibrary("NMS_Greedy_JNI");}

    /**
     *
     * @param dr the rectangle matrix where each row is a rectangle. Stored in col major.
     * @param ds the scores for reach rectangle.
     */
    public native void suppress(float[] dr, float[] ds, float overlap_thresh);

    public float[] get_dr_nms() {return dr_nms;}
    public float[] get_ds_nms() {return ds_nms;}

}
