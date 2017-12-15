package me.peyppicp.advance2;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class TestLSTM {

    public static void main(String[] args) throws IOException {

        int truncateReviewsToLength = 20;
        int vectorSize = 0;
        String wordVectorPath = "glove.twitter.27B.100d.txt";
        String trainDataPath = "EmojiSampleWithoutEmoji.txt";
        String labelDataPath = "EmojiSampleLabels.txt";
        String sampleFilePath = "EmojiSample.txt";
        String totalExamplesPath = "emoji_sample.txt";

        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("model-full01-4.txt");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(wordVectorPath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

        Random random = new Random(123456);
        List<String> labels = FileUtils.readLines(new File(labelDataPath), Charsets.UTF_8);
        List<String> testLines = FileUtils.readLines(new File(trainDataPath), Charsets.UTF_8);
        Collections.shuffle(testLines, random);
        Collections.shuffle(labels, random);
        labels = labels.subList(0, 1000);
        testLines = testLines.subList(0, 1000);

        int labelSize = labels.parallelStream().collect(Collectors.toSet()).size();
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.100d.txt");
        WordToIndex wordToIndex = new WordToIndex(sampleFilePath);

        int i = 0;
        for (String line : testLines) {
            int index = Integer.parseInt(labels.get(i).split(",")[0]);
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            List<String> tokenFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) {
                    tokenFiltered.add(t);
                }
            }
            int outputLength = Math.max(truncateReviewsToLength, tokenFiltered.size());
            INDArray features = Nd4j.create(1, vectorSize, outputLength);
            for (int j = 0; j < tokens.size() && j < truncateReviewsToLength; j++) {
                String token = tokens.get(j);
                INDArray vectorMatrix = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vectorMatrix);
            }

            INDArray output = multiLayerNetwork.output(features, false);
            int size = output.size(2);
            System.out.println();
            INDArray probabilitiesAtLastWord = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(size - 1));

            Number number = probabilitiesAtLastWord.maxNumber();
            long l = number.longValue();
            System.out.println(probabilitiesAtLastWord.toString());
            System.out.println("\n-------------------------------");
            System.out.println("review: " + line);
            System.out.println("Prefer:" + wordToIndex.getEmoji(index));
            System.out.println("label: " + probabilitiesAtLastWord.getDouble(1));
            i++;
            System.out.println();
        }
    }
}
