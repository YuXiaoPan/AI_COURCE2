package me.peyppicp.ai;

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
    private AtomicInteger correctTop3Number = new AtomicInteger();
    private AtomicInteger totalNumber = new AtomicInteger();

    private AtomicInteger correctEmojiTop1Number = new AtomicInteger();
    private AtomicInteger correctEmojiTop3Number = new AtomicInteger();
    private AtomicInteger totalEmojiNumber = new AtomicInteger();

    private PTBEvaluation() {
    }

    public void plusEmojiTop1Correct() {
        correctEmojiTop1Number.incrementAndGet();
    }

    public void plusEmojiTop3Correct() {
        correctEmojiTop3Number.incrementAndGet();
    }

    public void plusEmojiTotalNumber() {
        totalEmojiNumber.incrementAndGet();
    }

    public void plusEmojiTotalNumber(int size) {
        totalEmojiNumber.addAndGet(size);
    }

    public void plusTop1Correct() {
        correctTop1Number.incrementAndGet();
    }

    public void plusTop3Correct() {
        correctTop3Number.incrementAndGet();
    }

    public void plusTotalNumber() {
        totalNumber.incrementAndGet();
    }

    public void plusTotalNumber(int tokenSize) {
        totalNumber.addAndGet(tokenSize);
    }

    public float getCorrectEmojiTop1Rate() {
        return correctEmojiTop1Number.get() / totalEmojiNumber.floatValue();
    }

    public float getCorrectEmojiTop3Rate() {
        return correctEmojiTop3Number.get() / totalEmojiNumber.floatValue();
    }

    public float getCorrectTop1Rate() {
        return correctTop1Number.get() / totalNumber.floatValue();
    }

    public double getErrorTop1Rate() {
        return 1 - getCorrectTop1Rate();
    }

    public float getCorrectTop3Rate() {
        return correctTop3Number.get() / totalNumber.floatValue();
    }

    public double getErrorTop3Rate() {
        return 1 - getCorrectTop3Rate();
    }

    private static class Holder {
        private static PTBEvaluation ptbEvaluation = new PTBEvaluation();
    }

    public static PTBEvaluation getInstance() {
        return Holder.ptbEvaluation;
    }
}
