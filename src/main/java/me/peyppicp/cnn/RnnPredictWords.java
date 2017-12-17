package me.peyppicp.cnn;

import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.Utils;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class RnnPredictWords {

    public static final String UNKNOWN = "<unknown>";
    public static final String EMOJI = "<emoji>";
    public static final String END = "<end>";
    public static final String OUTPUT = "/home/peyppicp/output/";
    public static final String PREFIX = "/home/peyppicp/data/new/";
    private static final Logger log = LoggerFactory.getLogger(RnnPredictWords.class);

    public static void main(String[] args) throws IOException {
        File originData = new File(PREFIX + "more_standard_emoji_sample.txt");
        if (!originData.exists()) {
            preMain();
        }

        String prefix = "rnn";
        int truncateLength = 50;
        int batchSize = 64;
        int nEpochs = 50;
        List<String> samples = Utils.readLinesFromPath(originData.getCanonicalPath());
        WordToIndex wordToIndex = new WordToIndex(samples, 12500);

        RDataSetIterator rDataSetIterator = new RDataSetIterator(true, truncateLength, batchSize, samples, wordToIndex);
        RDataSetIterator tDataSetIterator = new RDataSetIterator(false, truncateLength, batchSize, samples, wordToIndex);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(new Random().nextInt(100))
                .updater(Updater.SGD)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(rDataSetIterator.inputColumns()).nOut(15000)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(15000).nOut(rDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));

        System.out.println("begin train");
        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(rDataSetIterator);
            Evaluation evaluate = multiLayerNetwork.evaluate(tDataSetIterator);
            System.out.println(evaluate.stats());
            rDataSetIterator.reset();
            ModelSerializer.writeModel(multiLayerNetwork, new File(OUTPUT + prefix + j + ".txt"), true);
        }
    }

    private static void preMain() throws IOException {
        String emojiSamples = PREFIX + "emoji_sample.txt";
        String input = PREFIX + "standard_emoji_samples.txt";
        Utils.processOriginalSamples(emojiSamples, input, true);
        List<String> lines = Utils.readLinesFromPath(input);
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + "glove.twitter.27B.100d.txt");
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        List<String> tempResults = new ArrayList<>();
        for (String line : lines) {
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            List<String> filteredTokens = new ArrayList<>(tokens.size());
            List<String> extractEmojis = EmojiParser.extractEmojis(line);
            for (String token : tokens) {
                if (word2Vec.hasWord(token)) {
                    filteredTokens.add(token);
                } else {
                    filteredTokens.add(UNKNOWN);
                }
            }
            if (!extractEmojis.isEmpty()) {
                filteredTokens.add(EMOJI);
            } else {
                filteredTokens.add(END);
            }
            StringBuilder stringBuilder = new StringBuilder();
            filteredTokens.forEach(s -> stringBuilder.append(s).append(" "));
            tempResults.add(stringBuilder.toString().trim());
        }

        WordToIndex wordToIndex = new WordToIndex(tempResults, 20000);
        List<String> finalResults = new ArrayList<>(tempResults.size());
        for (String tempResult : tempResults) {
            List<String> tokens = tokenizerFactory.create(tempResult).getTokens();
            List<Integer> indexes = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                indexes.add(wordToIndex.getWordIndex(token));
            }
            StringBuilder stringBuilder = new StringBuilder();
            indexes.forEach(s -> stringBuilder.append(s).append(" "));
            finalResults.add(stringBuilder.toString().trim());
        }
        String output = PREFIX + "more_standard_emoji_sample.txt";
        Utils.writeLineToPath(finalResults, output);
    }
}