package me.peyppicp.cnn;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.vdurmont.emoji.EmojiParser;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import me.peyppicp.Utils;
import me.peyppicp.advance2.EmojiToIndex;
import me.peyppicp.advance2.HibernateInfoRunner;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * @author YuXiao Pan
 * @date 2017/12/16
 * @email yuxiao.pan@kikatech.com
 * 根据整句输入预测emoji
 */
public class CNNDecideEmojiMain {

    private static final Logger log = LoggerFactory.getLogger(CNNDecideEmojiMain.class);
    private static ArrayListMultimap<String, SampleIndexPair> emojiToSamples = ArrayListMultimap.create();
//    public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
    public static final String PREFIX = "/home/panyuxiao/data/new/";
    public static final String OUTPUT = "/home/panyuxiao/output/";
//    public static final String PREFIX = "";
//    public static final String OUTPUT = "";

    public static void main(String[] args) throws IOException {
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(false)
                .setMaximumDeviceCache(10L * 1024L * 1024L * 1024L)
                .allowCrossDeviceAccess(true);

        String prefix = "cnn01";

        String sampleWithEmoji = PREFIX + "EmojiSample.txt";
        String samplePath = PREFIX + "EmojiSampleWithoutEmoji.txt";
        String sampleLabelPath = PREFIX + "EmojiSampleLabels.txt";
        String word2VecPath = PREFIX + "glove.twitter.27B.100d.txt";
        if (!(new File(sampleLabelPath).exists() && new File(samplePath).exists()
                && new File(sampleWithEmoji).exists() && new File(word2VecPath).exists())) {
            System.out.println("Begin process original samples.");
            Utils.processOriginalSamples(PREFIX + "emoji_sample.txt", sampleWithEmoji, false);
//            processOriginalSamples(new File(PREFIX + "emoji_sample.txt"));
//            processWord2Vec();
            System.out.println("Begin mark labels.");
            Utils.markLabels(sampleWithEmoji, sampleLabelPath);
            System.out.println("Begin remove emojis.");
            Utils.removeEmojis(sampleWithEmoji, samplePath);
        }

        ExecutorService executorService = Executors.newFixedThreadPool(2);
        List<String> samples = FileUtils.readLines(new File(samplePath), Charsets.UTF_8);
        List<String> sampleLabels = FileUtils.readLines(new File(sampleLabelPath), Charsets.UTF_8);
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(new File(word2VecPath));
        EmojiToIndex EmojiToIndex = new EmojiToIndex(sampleWithEmoji, 25);

        int batchSize = 100;
        int vectorSize = word2Vec.getWordVector(word2Vec.vocab().wordAtIndex(0)).length;
        int nEpochs = 5000;
        int truncateReviewsToLength = 64;
        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .regularization(true).l2(0.0001)
                .learningRate(0.01)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3 * cnnLayerFeatureMaps)
                        .nOut(EmojiToIndex.getOutComesNum())    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                .build();


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSetIterator trainIter = getDataSetIterator(true, word2Vec, batchSize, truncateReviewsToLength, rng, samples, sampleLabels, EmojiToIndex);
        DataSetIterator testIter = getDataSetIterator(false, word2Vec, batchSize, truncateReviewsToLength, rng, samples, sampleLabels, EmojiToIndex);

        net.setListeners(new StatsListener(statsStorage));
        log.info("Starting training");

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");
            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(testIter);
            System.out.println(evaluation.stats());
            trainIter = getDataSetIterator(true, word2Vec, batchSize, truncateReviewsToLength, rng, samples, sampleLabels, EmojiToIndex);
            testIter = getDataSetIterator(false, word2Vec, batchSize, truncateReviewsToLength, rng, samples, sampleLabels, EmojiToIndex);
            if (i % 10 == 0) {
                executorService.submit(new HibernateInfoRunner(i, net, trainIter, prefix, evaluation));
            }
        }
    }

    public static List<List<String>> getSamplesLineAndLabel(boolean isTrain, List<String> samples, List<String> sampleLabels) {
        List<String> sentences = new ArrayList<>();
        List<String> labelForSentences = new ArrayList<>();
        ImmutableList<List<String>> immutableList = ImmutableList.of(sentences, labelForSentences);
        Preconditions.checkArgument(samples.size() == sampleLabels.size());
        for (int i = 0; i < samples.size(); i++) {
            String s = EmojiParser.removeAllEmojis(samples.get(i)).trim().toLowerCase();
            List<String> indexes = ImmutableList.copyOf(sampleLabels.get(i).split(","));
            for (String index : indexes) {
                if (!index.contains(String.valueOf(-1))) {
                    emojiToSamples.put(index, new SampleIndexPair(s, index));
                }
            }
        }
        if (isTrain) {
            int maxCount = 5000;
            for (String index : emojiToSamples.keySet()) {
                List<SampleIndexPair> sampleIndexPairs = emojiToSamples.get(index);
                Collections.shuffle(sampleIndexPairs);
                int totalCount = sampleIndexPairs.size();
                for (int i = 0; i < Math.min(maxCount, totalCount); i++) {
                    sentences.add(sampleIndexPairs.get(i).getSample());
                    labelForSentences.add(sampleIndexPairs.get(i).getIndex());
                }
//                for (int i = 0; i < sampleIndexPairs.size(); i++) {
//                    sentences.add(sampleIndexPairs.get(i).getSample());
//                    labelForSentences.add(sampleIndexPairs.get(i).getIndex());
//                }
            }
        } else {
            int maxCount = 500;
            for (String index : emojiToSamples.keySet()) {
                List<SampleIndexPair> sampleIndexPairs = emojiToSamples.get(index);
                Collections.shuffle(sampleIndexPairs);
                int totalCount = sampleIndexPairs.size();
                for (int i = 0; i < Math.min(maxCount, totalCount); i++) {
                    sentences.add(sampleIndexPairs.get(i).getSample());
                    labelForSentences.add(sampleIndexPairs.get(i).getIndex());
//            }
                }
            }
        }
        return immutableList;
    }

    public static DataSetIterator getDataSetIterator(boolean isTrain, WordVectors wordVectors,
                                                     int miniBatchSize, int maxSentenceLength, Random random,
                                                     List<String> samples, List<String> sampleLabels,
                                                     EmojiToIndex EmojiToIndex) {
        List<List<String>> samplesLineAndLabel = getSamplesLineAndLabel(isTrain, samples, sampleLabels);
        CollectionLabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(samplesLineAndLabel.get(0),
                samplesLineAndLabel.get(1), random);
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

