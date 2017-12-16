package me.peyppicp.cnn;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.vdurmont.emoji.EmojiParser;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import me.peyppicp.advance2.WordToIndex;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author YuXiao Pan
 * @date 2017/12/16
 * @email yuxiao.pan@kikatech.com
 */
public class CNNMain {

    private static ArrayListMultimap<String, SampleIndexPair> emojiToSamples = ArrayListMultimap.create();
//    public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
//    public static final String PREFIX = "/home/panyuxiao/data/new/";
//    public static final String OUTPUT = "/home/panyuxiao/output/";
    public static final String PREFIX = "";
    public static final String OUTPUT = "";

    public static void main(String[] args) {

        String samplePath = PREFIX + "EmojiSampleWithoutEmoji.txt";
        String sampleLabelPath = PREFIX + "EmojiSampleLabels.txt";

        int batchSize = 64;
        int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int nepochs = 100;
        int truncateReviewsToLength = 64;
        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345);
    }

    private static DataSetIterator getDataSetIterator(boolean isTrain, WordVectors wordVectors,
                                                      int miniBatchSize, int maxSentenceLength, Random random,
                                                      List<String> samples, List<String> sampleLabels,
                                                      WordToIndex wordToIndex) {
        Preconditions.checkArgument(samples.size() == sampleLabels.size());
        for (int i = 0; i < samples.size(); i++) {
            String s = EmojiParser.removeAllEmojis(samples.get(i)).trim().toLowerCase();
            List<String> indexes = ImmutableList.copyOf(sampleLabels.get(i).split(","));
            for (String index : indexes) {
                emojiToSamples.put(index, new SampleIndexPair(s, index));
            }
        }
        ArrayList<String> sentences = new ArrayList<>();
        ArrayList<String> labelForSentences = new ArrayList<>();
        for (String index : emojiToSamples.keySet()) {
            List<SampleIndexPair> sampleIndexPairs = emojiToSamples.get(index);
            Stream<SampleIndexPair> beforeMapStream = sampleIndexPairs.parallelStream().distinct().limit(1000);
            sentences.addAll(beforeMapStream.map(SampleIndexPair::getSample).collect(Collectors.toList()));
            labelForSentences.addAll(beforeMapStream.map(SampleIndexPair::getIndex).collect(Collectors.toList()));
        }
        CollectionLabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labelForSentences, random);
        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(sentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(miniBatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class SampleIndexPair {
        private String sample;
        private String index;
    }
}
