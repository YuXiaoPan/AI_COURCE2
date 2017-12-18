package me.peyppicp.cnn;

import me.peyppicp.Utils;
import me.peyppicp.advance2.EmojiToIndex;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * @author YuXiao Pan
 * @date 2017/12/18
 * @email yuxiao.pan@kikatech.com
 */
public class RnnPredictEmoji {

        public static final String OUTPUT = "/home/peyppicp/output/";
        public static final String PREFIX = "/home/peyppicp/data/new/";
//    public static final String PREFIX = "/home/panyuxiao/data/new/";
//    public static final String OUTPUT = "/home/panyuxiao/output/";
//    public static final String PREFIX = "";
//    public static final String OUTPUT = "";

    public static void main(String[] args) throws IOException {
        String originSamples = PREFIX + "emoji_sample.txt";
        String emojiSampleWithEmoji = PREFIX + "EmojiSample.txt";
        String samplesPath = PREFIX + "EmojiSampleWithoutEmoji.txt";
        String sampleLabelPath = PREFIX + "EmojiSampleLabels.txt";

        System.out.println("Begin process original samples.");
        Utils.processOriginalSamples(originSamples, emojiSampleWithEmoji, false);
        System.out.println("Begin mark labels.");
        Utils.markLabels(emojiSampleWithEmoji, sampleLabelPath);
        System.out.println("Begin remove emojis.");
        Utils.removeEmojis(emojiSampleWithEmoji, samplesPath);

        List<String> samples = Utils.readLinesFromPath(samplesPath);
        List<String> sampleLabels = Utils.readLinesFromPath(sampleLabelPath);
        List<List<String>> samplesLineAndLabel = CNNDecideEmojiMain.getSamplesLineAndLabel(true, samples, sampleLabels);

        List<String> filteredSamples = samplesLineAndLabel.get(0);
        List<String> filteredSampleLabels = samplesLineAndLabel.get(1);

        Random random = new Random();
        Collections.shuffle(filteredSamples, random);
        Collections.shuffle(filteredSampleLabels, random);

        int batchSize = 200;
        int nEpochs = 100;
        int truncateLength = 30;

        EmojiToIndex emojiToIndex = new EmojiToIndex(PREFIX + "EmojiSample.txt", 25);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + "glove.twitter.27B.50d.txt");
        EmojiDataSetIterator train = new EmojiDataSetIterator(true,filteredSamples, filteredSampleLabels, emojiToIndex, batchSize
                , tokenizerFactory, word2Vec, truncateLength);
        EmojiDataSetIterator test = new EmojiDataSetIterator(false,filteredSamples, filteredSampleLabels, emojiToIndex, batchSize
                , tokenizerFactory, word2Vec, truncateLength);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(new Random().nextInt(100))
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(train.inputColumns()).nOut(50)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(50).nOut(train.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));
        for (int i = 0; i < nEpochs; i++) {
            multiLayerNetwork.fit(train);
            Evaluation evaluate = multiLayerNetwork.evaluate(test);
            System.out.println(evaluate);
            train.reset();
            ModelSerializer.writeModel(multiLayerNetwork, OUTPUT + "predict" + i + ".txt", true);
        }
    }
}
