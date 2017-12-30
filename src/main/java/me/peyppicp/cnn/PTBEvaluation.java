package me.peyppicp.cnn;

import lombok.Data;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
@Data
public class PTBEvaluation {

    private AtomicInteger correctTop1Number = new AtomicInteger();
    private AtomicInteger totalTop1Number = new AtomicInteger();
    private AtomicInteger errorTop1Number = new AtomicInteger();

    private AtomicInteger correctTop3Number = new AtomicInteger();
    private AtomicInteger totalTop3Number = new AtomicInteger();
    private AtomicInteger errorTop3Number = new AtomicInteger();

    private PTBEvaluation() {

    }

    public void plusTop1Current() {
        correctTop1Number.incrementAndGet();
        totalTop1Number.incrementAndGet();
    }

    public void plusTop1Error() {
        errorTop1Number.incrementAndGet();
        totalTop1Number.incrementAndGet();
    }

    public void plusTop3Current() {
        correctTop3Number.incrementAndGet();
        totalTop3Number.incrementAndGet();
    }

    public void plusTop3Error() {
        errorTop3Number.incrementAndGet();
        totalTop3Number.incrementAndGet();
    }

    public double getCorrectTop1Rate() {
        return correctTop1Number.get() / totalTop1Number.doubleValue();
    }

    public double getErrorTop1Rate() {
        return errorTop1Number.get() / totalTop1Number.doubleValue();
    }

    public double getCorrectTop3Rate() {
        return correctTop3Number.get() / totalTop3Number.doubleValue();
    }

    public double getErrorTop3Rate() {
        return errorTop3Number.get() / totalTop3Number.doubleValue();
    }

    private static class Holder {
        private static PTBEvaluation ptbEvaluation = new PTBEvaluation();
    }

    public static PTBEvaluation getInstance() {
        return Holder.ptbEvaluation;
    }
}
