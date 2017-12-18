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
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * @author YuXiao Pan
 * @date 2017/12/18
 * @email yuxiao.pan@kikatech.com
 */
public class RnnPredictEmoji {

    public static void main(String[] args) throws IOException {
        String samplesPath = "EmojiSampleWithoutEmoji.txt";
        String sampleLabelPath = "EmojiSampleLabels.txt";
        List<String> samples = Utils.readLinesFromPath(samplesPath);
        List<String> sampleLabels = Utils.readLinesFromPath(sampleLabelPath);
        List<List<String>> samplesLineAndLabel = CNNDecideEmojiMain.getSamplesLineAndLabel(true, samples, sampleLabels);

        List<String> filteredSamples = samplesLineAndLabel.get(0);
        List<String> filteredSampleLabels = samplesLineAndLabel.get(1);

        int batchSize = 128;
        int nEpochs = 10;
        int truncateLength = 30;

        EmojiToIndex emojiToIndex = new EmojiToIndex("EmojiSample.txt", 25);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.100d.txt");
        EmojiDataSetIterator train = new EmojiDataSetIterator(filteredSamples, filteredSampleLabels, emojiToIndex, batchSize
                , tokenizerFactory, word2Vec, truncateLength);
        EmojiDataSetIterator test = new EmojiDataSetIterator(filteredSamples, filteredSampleLabels, emojiToIndex, batchSize
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
                .layer(0, new GravesLSTM.Builder().nIn(train.inputColumns()).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(train.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        multiLayerNetwork.init();

        for (int i = 0; i < nEpochs; i++) {
            multiLayerNetwork.fit(train);
            Evaluation evaluate = multiLayerNetwork.evaluate(test);
            System.out.println(evaluate);
            train.reset();
            ModelSerializer.writeModel(multiLayerNetwork, "predict" + i + ".txt", true);
        }
    }
}
