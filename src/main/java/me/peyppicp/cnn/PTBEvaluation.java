package me.peyppicp.cnn;

import lombok.Data;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
@Data
public class PTBEvaluation {

    private int tpNumber;
    private int fpNumber;
    private int fnNumber;
    private int tnNumber;

    public void tpPlusOne() {
        tpNumber++;
    }

    public void fpPlusOne() {
        fpNumber++;
    }

    public void fnPlusOne() {
        fnNumber++;
    }

    public void tnPlusOne() {
        tnNumber++;
    }

    public double precision() {
        return tpNumber / ((tpNumber + fpNumber) * 1.0d);
    }

//    public double
}
