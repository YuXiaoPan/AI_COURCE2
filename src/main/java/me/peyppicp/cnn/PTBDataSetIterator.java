package me.peyppicp.cnn;

import com.google.common.collect.Lists;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class PTBDataSetIterator implements DataSetIterator {

    //    private final int linesToReadPerBatch;
    private final int batchSize;
    private final int truncateLength;
    private List<String> tokens;
    private int cursor = 0;
    private final WordToIndex wordToIndex;
    private final WordVectors wordVectors;
    private int vectorSize;
    private int fetchTokenSize;
    private final int numberSteps;

    public PTBDataSetIterator(boolean isTrain, int truncateLength, int batchSize, int numberSteps,
                              List<String> samples, WordToIndex wordToIndex,
                              WordVectors wordVectors) {
//        this.linesToReadPerBatch = linesToReadPerBatch;
        this.tokens = Lists.newArrayList();
        this.truncateLength = truncateLength;
        DefaultTokenizerFactory defaultTokenizerFactory = new DefaultTokenizerFactory();
        tokens = defaultTokenizerFactory.create(samples.get(0)).getTokens();
        if (!isTrain) {
            tokens = tokens.subList(0, 100);
        }
        this.batchSize = batchSize;
        this.wordToIndex = wordToIndex;
        this.wordVectors = wordVectors;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.numberSteps = numberSteps;
        this.fetchTokenSize = batchSize * numberSteps;

    }

    @Override
    public DataSet next(int batchSize) {
        if (cursor >= tokens.size()) {
            throw new RuntimeException();
        }

        int maxWordListSize = 0;
        int maxWordsSize = 0;
        List<List<String>> words = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            List<String> temp = new ArrayList<>();
            for (int j = 0; cursor < tokens.size() && j < numberSteps; cursor++, j++) {
                temp.add(tokens.get(cursor));
            }
            if (!temp.isEmpty()) {
                words.add(temp);
                maxWordsSize = Math.max(maxWordsSize, temp.size());
            }
        }

        maxWordListSize = Math.min(batchSize, words.size());
        INDArray input = Nd4j.create(new int[]{maxWordListSize,
                vectorSize, maxWordsSize}, 'f');
        INDArray labels = Nd4j.create(new int[]{maxWordListSize,
                wordToIndex.getTotalWordsCount(), maxWordsSize}, 'f');

        for (int i = 0; i < words.size(); i++) {
            List<String> firstBatch = words.get(i);
            String currentToken = firstBatch.get(0);
            INDArray currentVector = wordVectors.getWordVectorMatrix(currentToken);
            for (int j = 1, timeStep = 0; j < firstBatch.size(); j++, timeStep++) {
                String nextToken = firstBatch.get(j);
                input.put(new INDArrayIndex[]{NDArrayIndex.point(i),
                        NDArrayIndex.all(), NDArrayIndex.point(timeStep)}, currentVector);
                labels.putScalar(new int[]{i, wordToIndex.getWordIndex(nextToken), timeStep}, 1.0);
                currentVector = wordVectors.getWordVectorMatrix(nextToken);
                currentToken = nextToken;
            }
        }
        return new DataSet(input, labels);
    }

    @Override
    public int totalExamples() {
        return tokens.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
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
        return totalExamples();
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
        return cursor < tokens.size();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
