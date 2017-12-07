package me.peyppicp.method2;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class SampleDataSetIterator implements DataSetIterator {


    private final boolean isTest;

    private int batchSize;
    private int vectorSize;
    private String samplePath;
    private String wordVectorPath;
    private int cursor = 0;
    private int totalSize = 0;
    private List<String> totalSamples;
    private WordToIndex wordToIndex;
    private final TokenizerFactory tokenizerFactory;
    private WordVectors wordVectors;
    private int maxLength;

    public SampleDataSetIterator(int batchSize, boolean isTest, String samplePath, String wordVectorPath) throws IOException {
        this.batchSize = batchSize;
        this.isTest = isTest;
        this.samplePath = samplePath;
        this.wordVectorPath = wordVectorPath;
        this.wordToIndex = new WordToIndex(samplePath, wordVectorPath);
        this.wordVectors = wordToIndex.getWordVectors();
        this.tokenizerFactory = wordToIndex.getTokenizerFactory();
        this.totalSamples = FileUtils.readLines(new File(samplePath), Charsets.UTF_8);
        this.totalSize = totalSamples.size();
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    }

    public DataSet next(int batchSize) {
        List<String> batchSizeSamples = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize && cursor < totalSize; i++, cursor++) {
            String temp = totalSamples.get(cursor);
            batchSizeSamples.add(temp);
        }

        maxLength = 0;
        for (String sample : batchSizeSamples) {

        }
        return null;
    }

    public int totalExamples() {
        return totalSamples.size();
    }

    public int inputColumns() {
        return vectorSize;
    }

    public int totalOutcomes() {
        return wordToIndex.totalIndexNum();
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }

    public void reset() {
        this.cursor = 0;
    }

    public int batch() {
        return batchSize;
    }

    public int cursor() {
        return cursor;
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public List<String> getLabels() {
        return wordToIndex.totalLabels();
    }

    public boolean hasNext() {
        return cursor < numExamples();
    }

    public DataSet next() {
        return next(batchSize);
    }
}
