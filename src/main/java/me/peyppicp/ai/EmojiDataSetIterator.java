package me.peyppicp.ai;

import lombok.Data;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
@Data
public class EmojiDataSetIterator implements DataSetIterator {

    private int cursor;
    private List<String> samples;
    private List<Integer> sampleLabels;
    private final EmojiToIndex emojiToIndex;
    private final int batchSize;
    private final TokenizerFactory tokenizerFactory;
    private final WordVectors wordVectors;
    private final int vectorSize;
    private final int truncateLength;

    public EmojiDataSetIterator(boolean isTrain, List<String> samples, List<String> sampleLabels, EmojiToIndex emojiToIndex,
                                int batchSize, TokenizerFactory tokenizerFactory, WordVectors wordVectors,
                                int truncateLength) {
        this.samples = samples;
        this.sampleLabels = sampleLabels.stream().map(Integer::parseInt).collect(Collectors.toList());
        this.emojiToIndex = emojiToIndex;
        this.batchSize = batchSize;
        this.tokenizerFactory = tokenizerFactory;
        this.wordVectors = wordVectors;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.truncateLength = truncateLength;
        if (!isTrain) {
            this.samples = this.samples.subList(0, 5000);
            this.sampleLabels = this.sampleLabels.subList(0, 5000);
        }
    }

    public DataSet next(int batchSize) {
        if (cursor >= samples.size()) {
            throw new NoSuchElementException();
        }
        List<String> reviews = new ArrayList<>(batchSize);
        int[] labelInts = new int[batchSize];
        for (int i = 0; i < batchSize && cursor < totalExamples(); i++, cursor++) {
            String line = samples.get(cursor);
            reviews.add(line);
            labelInts[i] = sampleLabels.get(i);
        }
        List<List<String>> allTokens = new ArrayList<>();
        int maxLength = 0;
        for (String line : reviews) {
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            List<String> tokenFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) {
                    tokenFiltered.add(t);
                }
            }
            allTokens.add(tokenFiltered);
            maxLength = Math.max(maxLength, tokenFiltered.size());
        }

        if (maxLength > truncateLength) {
            maxLength = truncateLength;
        }
        INDArray features = Nd4j.create(new int[]{reviews.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{reviews.size(), emojiToIndex.totalOutputNumber(), maxLength}, 'f');
        INDArray featureMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelMask = Nd4j.zeros(reviews.size(), maxLength);

//        int[] ints = new int[maxLength];
        int[] ints = new int[2];
        for (int i = 0; i < reviews.size(); i++) {
            ints[0] = i;
            List<String> tokens = allTokens.get(i);
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i),
                        NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
                ints[1] = j;
                featureMask.putScalar(ints, 1.0);
            }
            int index = labelInts[i];
            int lastIndex = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, index, lastIndex - 1}, 1.0);
            labelMask.putScalar(new int[]{i, lastIndex - 1}, 1.0);
        }
        return new DataSet(features, labels, featureMask, labelMask);
    }

    public int totalExamples() {
        return samples.size();
    }

    public int inputColumns() {
        return vectorSize;
    }

    public int totalOutcomes() {
        return emojiToIndex.totalOutputNumber();
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }

    public void reset() {
        this.cursor = 0;
        Random random = new Random();
        Collections.shuffle(this.samples, random);
        Collections.shuffle(this.sampleLabels, random);
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
        return emojiToIndex.totalLabels();
    }

    public boolean hasNext() {
        return cursor < numExamples();
    }

    public DataSet next() {
        return next(batchSize);
    }
}
