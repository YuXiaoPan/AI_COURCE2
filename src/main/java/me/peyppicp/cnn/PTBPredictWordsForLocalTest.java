package me.peyppicp.cnn;

import com.google.common.collect.Lists;
import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.Utils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
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
public class PTBPredictWordsForLocalTest {

    public static final String UNKNOWN = "<unknown>";
    public static final String EMOJI = "<emoji>";
    public static final String END = "<end>";
    //        public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
//    public static final String PREFIX = "/home/panyuxiao/data/new/";
//    public static final String OUTPUT = "/home/panyuxiao/output/";
        public static final String PREFIX = "";
    public static final String OUTPUT = "";
    private static final int limitNum = 15000;
    private static final Logger log = LoggerFactory.getLogger(PTBPredictWordsForLocalTest.class);

    public static void main(String[] args) throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        File originData = new File(PREFIX + "test.txt");
        if (!originData.exists()) {
            preMain();
            preForWord2Vec();
        }

        String prefix = "peyppicp";
        int truncateLength = 30;
        int batchSize = 4;
        int nEpochs = 500;
        int numberSteps = 5;
        List<String> samples = Utils.readLinesFromPath(originData.getCanonicalPath());
        WordToIndex wordToIndex = new WordToIndex(samples, limitNum);
//        WordVectors word2Vec = rebuildWord2Vec(Utils.readLinesFromPath(PREFIX + "more_standard_emoji_sample_for_word2vec.txt"));
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + "sub.word2vec.txt");
        PTBDataSetIterator rDataSetIterator = new PTBDataSetIterator(true, truncateLength, batchSize,
                numberSteps, samples, wordToIndex, word2Vec);
//        PTBDataSetIterator tDataSetIterator = new PTBDataSetIterator(false, truncateLength, batchSize,
//                numberSteps, Utils.readLinesFromPath("testForTest.txt"), wordToIndex, word2Vec);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .seed(new Random().nextInt(100))
                .updater(Updater.ADAM)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.001)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(rDataSetIterator.inputColumns()).nOut(75)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(75).nOut(rDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();

//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        multiLayerNetwork.init();

        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        RnnTest rnnTest = new RnnTest(word2Vec, wordToIndex, tokenizerFactory, multiLayerNetwork, null);

        System.out.println("begin train");
        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(rDataSetIterator);
//            Evaluation evaluate = multiLayerNetwork.evaluate(tDataSetIterator);
//            System.out.println(evaluate.stats());
            rDataSetIterator.reset();
//            if (j % 10 == 0) {
//                Evaluation evaluate = multiLayerNetwork.evaluate(tDataSetIterator);
//                System.out.println(evaluate);
//            }
//            ModelSerializer.writeModel(multiLayerNetwork, new File(OUTPUT + prefix + j + ".txt"), true);
        rnnTest.generateTokensFromStr("welcome to new york");
        }
    }

    private static void preForWord2Vec() throws IOException {
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

        WordToIndex wordToIndex = new WordToIndex(tempResults, limitNum);
        List<String> finalResults = new ArrayList<>(tempResults.size());
        for (String tempResult : tempResults) {
            StringBuilder stringBuilder = new StringBuilder();
            List<String> tokens = tokenizerFactory.create(tempResult).getTokens();
            List<String> indexes = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                if (wordToIndex.getWordIndex(token) != wordToIndex.getWordIndex(UNKNOWN)) {
                    indexes.add(token);
                } else {
                    indexes.add(UNKNOWN);
                }
            }
            indexes.forEach(s -> stringBuilder.append(s).append(" "));
            finalResults.add(stringBuilder.toString().trim());
        }
        String output = PREFIX + "more_standard_emoji_sample_for_word2vec.txt";
        Utils.writeLineToPath(finalResults, output);
    }

    private static WordVectors rebuildWord2Vec(List<String> originSamples) throws IOException {
        File subWord2VecFile = new File(PREFIX + "sub.word2vec.txt");
        if (!subWord2VecFile.exists()) {
            int minWordFrequency = 5;
            int iterations = 1;
            int layerSize = 50;
            int seed = 3543;
            int windowSize = 10;

            CollectionSentenceIterator iterator = new CollectionSentenceIterator(originSamples);
            DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
            Word2Vec vec = new Word2Vec.Builder()
                    .minWordFrequency(minWordFrequency)
                    .iterations(iterations)
                    .layerSize(layerSize)
                    .seed(seed)
                    .windowSize(windowSize)
                    .iterate(iterator)
                    .tokenizerFactory(tokenizerFactory)
                    .build();
            vec.fit();
            WordVectorSerializer.writeWordVectors(vec.lookupTable(), PREFIX + "sub.word2vec.txt");
            return vec;
        }
        return WordVectorSerializer.readWord2VecModel(subWord2VecFile);
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

        WordToIndex wordToIndex = new WordToIndex(tempResults, limitNum);
//        List<String> finalResults = new ArrayList<>(tempResults.size());
        StringBuilder stringBuilder = new StringBuilder();
        for (String tempResult : tempResults) {
            List<String> tokens = tokenizerFactory.create(tempResult).getTokens();
            List<String> indexes = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                if (wordToIndex.getWordIndex(token) != wordToIndex.getWordIndex(UNKNOWN)) {
                    indexes.add(token);
                } else {
                    indexes.add(UNKNOWN);
                }
            }
            indexes.forEach(s -> stringBuilder.append(s).append(" "));
//            finalResults.add(stringBuilder.toString().trim());
        }
        List<String> data = Lists.newArrayList();
        data.add(stringBuilder.toString().trim());
        String output = PREFIX + "more_standard_emoji_sample.txt";
        Utils.writeLineToPath(data, output);
    }
}
