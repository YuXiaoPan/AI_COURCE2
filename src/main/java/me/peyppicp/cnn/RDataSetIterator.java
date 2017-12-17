package me.peyppicp.cnn;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class RDataSetIterator implements DataSetIterator {

    //    private final int linesToReadPerBatch;
    private final int batchSize;
    private final int truncateLength;
    private List<String> samples;
    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;
    private final WordToIndex wordToIndex;

    public RDataSetIterator(boolean isTrain, int truncateLength, int batchSize,
                            List<String> samples, WordToIndex wordToIndex) {
//        this.linesToReadPerBatch = linesToReadPerBatch;
        this.truncateLength = truncateLength;
        this.samples = samples;
        this.batchSize = batchSize;
        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        this.wordToIndex = wordToIndex;
        if (isTrain) {
            samples = samples.subList(0, 5000);
        }
        Collections.shuffle(samples);
    }

    @Override
    public DataSet next(int batchSize) {
        if (cursor >= samples.size()) {
            throw new RuntimeException();
        }

        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < batchSize && cursor < samples.size(); i++, cursor++) {
            String sample = samples.get(cursor);
            List<String> tokens = Arrays.stream(sample.split(" ")).collect(Collectors.toList());
            tokens.parallelStream().map(Integer::parseInt).forEachOrdered(indexes::add);
//            indexes.add(tokens.parallelStream().map(Integer::parseInt).collect(Collectors.toList()));
        }

        int maxWordsSize = Math.min(truncateLength, indexes.size());
        INDArray input = Nd4j.create(new int[]{batchSize,
                wordToIndex.getTotalWordsCount(), maxWordsSize}, 'f');
        INDArray labels = Nd4j.create(new int[]{batchSize,
                wordToIndex.getTotalWordsCount(), maxWordsSize}, 'f');

        for (int i = 0; i < indexes.size(); i++) {
            Integer currentIndex = indexes.get(i);
            int timeStep = 0;
            for (int j = i + 1; j < maxWordsSize; j++, timeStep++) {
                Integer nextIndex = indexes.get(j);
                input.putScalar(new int[]{i, currentIndex, timeStep}, 1.0);
                labels.putScalar(new int[]{i, nextIndex, timeStep}, 1.0);
            }

        }
        return new DataSet(input, labels);
    }

    @Override
    public int totalExamples() {
        return samples.size();
    }

    @Override
    public int inputColumns() {
        return wordToIndex.getTotalWordsCount();
    }

    @Override
    public int totalOutcomes() {
        return wordToIndex.getTotalWordsCount();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return wordToIndex.getLabels();
    }

    @Override
    public boolean hasNext() {
        return cursor < samples.size();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
